"""One fixed DBRL-versus-temporal reconstruction comparison; development evidence only."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from src.scientific_validation.reconstruction.runtime import new_output, require, sha, write
from src.scientific_validation.dbrl_decision.inputs import prepare, MANIFEST
from src.scientific_validation.dbrl_decision.model import (
    paired_models, state_hash, schedule, backward_gate, train_arm, extract, sync)
from src.scientific_validation.dbrl_decision.statistics import (
    pca_fit, pca_apply, fit_readouts, predict, per_target, macro, day_bootstrap,
    decision, summaries)

PROTOCOL='bre_dbrl_decision_v1'
SEEDS=(20261041,20261042,20261043)
COHORTS=('uniform','event_enriched')


def config_load(project_root):
    # Preserve the existing config loading convention; do not rewrite YAML.
    with open('configs/config.yaml') as f: config=yaml.safe_load(f)
    require(isinstance(config,dict),'Configuration must be a mapping')
    if project_root: config['project_root_dir']=project_root
    return copy.deepcopy(config)


def check_config(config):
    p=config['preprocessing'];d=config['hdeg']['dbrl']
    require(p['dataset_name']=='BRE' and not p.get('merge_bre_cu',False),'This frozen comparison requires BRE only')
    require(p['window_size']==30 and p.get('downsample_freq')=='1s','This protocol requires 30-window / 1s inputs')
    for key,value in dict(hidden_dim=64,embedding_dim=64,gru_layers=1,graph_heads=1).items():
        require(d.get(key,value)==value,'DBRL setting conflicts with fixed protocol: '+key)
    # Prior hbf/mbai/training overwrite, batch and step settings do not govern this experiment.


def checkpoint(path,model,seed,arm,stage,contract_hash):
    torch.save(dict(protocol=PROTOCOL,seed=seed,arm=arm,stage=stage,
        contract_sha256=contract_hash,model={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)


def source_dependencies():
    paths={Path(__file__).resolve(),Path('configs/config.yaml').resolve()}
    for name,module in tuple(sys.modules.items()):
        if name.startswith('src.') and getattr(module,'__file__',None):
            p=Path(module.__file__).resolve()
            if p.suffix=='.py': paths.add(p)
    return {str(p):sha(p) for p in paths}


def measure_latency(model,data,device):
    model.eval(); x=torch.from_numpy(data['X'][:8]).to(device)
    with torch.no_grad():
        for _ in range(3): model.encoder(x)
        sync(device);start=time.perf_counter()
        for _ in range(20): model.encoder(x)
        sync(device)
    return dict(encoder_ms_per_window=(time.perf_counter()-start)*1000/(20*len(x)),
                batch_size=len(x),timed_batches=20,warmup_batches=3)


def evaluate_features(name,features,data,desc,primary,final,out,report,score_bank,do_pca=True):
    # Neither representation projection nor readout fitting accesses evaluation labels.
    result={}
    views={'native':features}
    if do_pca:
        p=pca_fit(features['update'])
        np.savez_compressed(out/(name+'_pca.npz'),**p)
        views['pca320']={key:pca_apply(v,p) for key,v in features.items()}
        result['pca_rank']=p['rank'];result['pca_retained']=min(p['rank'],320)
    for view,arrays in views.items():
        params=fit_readouts(arrays['fit'],data['fit']['y'],data['fit']['valid'])
        np.savez_compressed(out/(name+'_'+view+'_readouts.npz'),
            **{str(j)+'_'+key:value for j,p in params.items() for key,value in p.items()})
        result[view]={}
        for cohort in COHORTS:
            scores=predict(arrays[cohort],params);b=data[cohort]
            np.savez_compressed(out/(name+'_'+view+'_'+cohort+'_scores.npz'),scores=scores,
                source_start=b['source_start'],source_stop=b['source_stop'],timestamp_end=b['timestamp_end'])
            result[view][cohort]=dict(per_target=per_target(b['y'],b['valid'],scores,desc),
                primary_ap=macro(b['y'],b['valid'],scores,primary),
                final_state_balanced_accuracy=macro(b['y'],b['valid'],scores,final,'balanced_accuracy'))
            if cohort=='event_enriched': score_bank[(name,view)]=scores
    report['readouts'][name]=result
    print('Readouts complete:',name,flush=True)


def run(args,config,out,report):
    check_config(config)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    require(device.type!='cuda' or torch.cuda.is_available(),'CUDA requested but unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    report['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),
        numpy=np.__version__,device=str(device),cuda=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        deterministic_algorithms=True,torch_threads=1,blas_threads=1)
    contract=dict(protocol=PROTOCOL,mode='verification_only' if args.verify_only else 'comparison',
        prepared_manifest_sha256=MANIFEST,seeds=list(SEEDS),updates_per_arm=1024,batch_size=8,
        optimizer=dict(name='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clipping=None),
        trainable_arms=['temporal','dbrl'],initial_controls=True,
        architecture=dict(hidden_dim=64,embedding_dim=64,gru_layers=1,dropout=0.,graph_heads=1,
            graph='dense endpoint-eligible residual context',decoder='flatten, Linear(N*64,64), GELU, Linear(64,W*N)'),
        loss='Per-window valid-value reconstruction MSE; window mean; no upper hierarchy or auxiliary objective',
        readout=dict(ridge=1.,threshold=.5,standardization='fit-only',projection='update-only full-SVD PCA, centered, no whitening, width320'),
        thresholds=dict(primary_ap_margin=.03,final_state_noninferiority=.02,maximum_undefined_bootstrap_fraction=.05),
        bootstrap=dict(replicates=1000,seed=20261044,cluster='source calendar date of window endpoint',
            pairing='same resampled days across methods and seeds; average seed differences, not independent sample pooling'),
        selection_report_sha256=sha(args.selection_report),evaluation_report_sha256=sha(args.evaluation_report),
        limitations=['All evaluation periods have informed development','States/transitions are input-derived proxies, not human activities',
            'Reconstruction errors are not validated anomaly scores','Few calendar days limit uncertainty estimates',
            'Additional graph parameters are not a parameter-matched topology intervention',
            'Exact raw rules are a reference ceiling, not a learning baseline to outperform'])
    write(out/'experiment_contract.json',contract);contract_hash=sha(out/'experiment_contract.json')
    deps=source_dependencies(); report['source_hashes']=dict(deps)
    data,ds,input_deps,desc,primary,final,coverage=prepare(args,out)
    deps.update(input_deps); report['input_dependencies']=input_deps
    report.update(coverage=coverage,readouts={},training={},step_zero={},optimizer_steps=0)
    n=data['update']['X'].shape[2]; require(n==62,'Frozen BRE channel count differs')
    def unchanged():
        for dataset in ds.values():dataset.verify_unchanged()
        require(all(sha(path)==digest for path,digest in deps.items()),'Consumed source or parent changed')
    # Gate ALL seeds before any arm trains. Backpropagation does not step an optimizer.
    for seed in SEEDS:
        arms=paired_models(seed,n)
        report['step_zero'][str(seed)]={}
        for arm,model in arms.items():
            model.to(device)
            loss=backward_gate(model,data['update'],device)
            report['step_zero'][str(seed)][arm]=dict(loss=loss,state_sha256=state_hash(model),
                encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                decoder_parameters=sum(p.numel() for p in model.decoder.parameters()))
            model.cpu()
        del arms
    write(out/'step_zero.json',report['step_zero'])
    unchanged()
    if args.verify_only:
        report.update(status='VERIFIED_ONLY',decision=dict(outcome='NOT_RUN',scope='No optimizer steps or scientific comparison'))
        return
    score_bank={}
    # All labels and inputs are frozen; exact rules are evaluated without fitting.
    for cohort in COHORTS:
        b=data[cohort]
        report['readouts']['exact_rules_'+cohort]=per_target(b['y'],b['valid'],b['y'],desc)
    raw={key:b['X'].reshape(len(b['X']),-1) for key,b in data.items()}
    evaluate_features('raw',raw,data,desc,primary,final,out,report,score_bank)
    del raw
    simple={key:summaries(b['values'],b['status']) for key,b in data.items()}
    evaluate_features('summary',simple,data,desc,primary,final,out,report,score_bank,False)
    del simple
    for seed in SEEDS:
        ids=schedule(seed);np.save(out/('schedule_'+str(seed)+'.npy'),ids,allow_pickle=False)
        arms=paired_models(seed,n)
        for arm,model in arms.items():
            model.to(device);tag=str(seed)+'_'+arm
            require(state_hash(model)==report['step_zero'][str(seed)][arm]['state_sha256'],'Seed recreation differs')
            checkpoint(out/(tag+'_initial.pt'),model,seed,arm,'initial',contract_hash)
            for stage in ('initial','trained'):
                if stage=='trained':
                    print('Training',tag,'1024 updates',flush=True)
                    def progress(step):
                        report['optimizer_steps']+=1
                        if step%128==0:print(tag,step,'/1024',flush=True)
                    result=train_arm(model,data['update'],ids,device,progress)
                    result.update(measure_latency(model,data['update'],device))
                    result['cpu_process_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
                    result['cpu_memory_scope']='Process lifetime high-water RSS on Linux; not isolated per-arm memory'
                    result['encoder_parameters']=sum(p.numel() for p in model.encoder.parameters())
                    result['decoder_parameters']=sum(p.numel() for p in model.decoder.parameters())
                    report['training'][tag]=result
                    checkpoint(out/(tag+'_trained.pt'),model,seed,arm,'trained',contract_hash)
                before=state_hash(model);features={};errors={};extract_times={}
                for subset,b in data.items():
                    features[subset],errors[subset],extract_times[subset]=extract(model,b,device)
                require(before==state_hash(model),'Evaluation changed model state')
                name=str(seed)+'_'+stage+'_'+arm
                np.savez_compressed(out/(name+'_reconstruction.npz'),**errors)
                report.setdefault('reconstruction',{})[name]={k:dict(mse=float(e.mean()),
                    representation_variance=float(features[k].astype(float).var(0).mean()),
                    extraction_seconds=extract_times[k]) for k,e in errors.items()}
                evaluate_features(name,features,data,desc,primary,final,out,report,score_bank)
                del features
            model.cpu()
        del arms
        unchanged()
    b=data['event_enriched'];names=('trained_dbrl','trained_temporal','initial_dbrl')
    scores={name:[score_bank[(str(seed)+'_'+name,'native')] for seed in SEEDS] for name in names}
    uncertainty,draws=day_bootstrap(b['y'],b['valid'],scores,b['timestamp_end'],primary)
    np.savez_compressed(out/'paired_day_bootstrap.npz',**draws)
    write(out/'uncertainty.json',uncertainty)
    native={name:[macro(b['y'],b['valid'],s,primary) for s in runs] for name,runs in scores.items()}
    compact={name:[macro(b['y'],b['valid'],score_bank[(str(seed)+'_'+name,'pca320')],primary)
                   for seed in SEEDS] for name in ('trained_dbrl','trained_temporal')}
    compact['raw']=[macro(b['y'],b['valid'],score_bank[('raw','pca320')],primary)]
    state={name:[macro(b['y'],b['valid'],s,final,'balanced_accuracy') for s in runs] for name,runs in scores.items()}
    resource_ok=len(report['training'])==6 and all(v['seconds']>0 and v['encoder_ms_per_window']>0
        and v['cpu_process_peak_rss_bytes']>0 for v in report['training'].values())
    report['decision_inputs']=dict(native=native,compact=compact,final_state=state,uncertainty=uncertainty)
    report['decision']=decision(native,compact,state,uncertainty,resource_ok)
    require(report['optimizer_steps']==6144,'Optimizer budget differs')
    unchanged();report['status']='COMPLETE'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir')
    p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--selection_report',type=Path,required=True)
    p.add_argument('--evaluation_report',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--verify_only',action='store_true',help='Verify real inputs and all seed backward gates; zero optimizer updates')
    args=p.parse_args();config=config_load(args.project_root_dir)
    out=new_output(config,'bre_dbrl_decision');write(out/'configuration_snapshot.json',config)
    report=dict(protocol=PROTOCOL,status='RUNNING',optimizer_steps=0)
    try:
        with threadpool_limits(limits=1):run(args,config,out,report)
    except Exception as e:
        report.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc())
    # Saved NaN bootstrap entries denote undefined replicates, never zero AP.
    report['output_hashes']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}
    write(out/'decision_report.json',report)
    print('DBRL decision:',report['status']);print('Report:',out/'decision_report.json')
    if report['status']=='BLOCKED':print(report['error']);raise SystemExit(2)
    if report.get('decision'):print('Decision:',report['decision']['outcome'])


if __name__=='__main__':main()
