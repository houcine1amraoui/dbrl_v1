"""Verify exact frozen selections against prepared data before any optimization."""
import json
from pathlib import Path
import numpy as np
from src.data_preparation.bre_windows import BREWindowDataset
from src.scientific_validation.reconstruction.data import window_batch
from src.scientific_validation.reconstruction.runtime import require, sha, write
from src.scientific_validation.behavior_preservation.core import targets, support, eligible
from src.scientific_validation.behavior_preservation.event_manifest import load_selection

MANIFEST='3d34c85214cbd1c6a8cda24b57b7c4f8df8abc5820877c80d13831433b12e3e2'
TRAIN_SELECTION='23b91d1d4e5b23346501dd6cee537dd917b5d3114ca2190da23cd38177c63768'
EVAL_SELECTION='77ab5810f5fdf851f7774cea28ecdc39241cf56cac2b92aad5e38b765cf506b2'
HISTORICAL_CHECKPOINT='98dd287013098294cb220a869b11a9f22240b3db585bce99b9d292ff5109e67e'
ORIGINAL_SELECTION='e29ae4933edf1ee26b560726d637367a8a4e36cd1f934c83c8835e4d2d733e4a'


def read(path):
    require(Path(path).is_file(),'Expected a file: '+str(path))
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k] for k in a.files}


def exact_rules(values,status,channels):
    """Independent loop oracle for the vectorized target routine."""
    y=np.zeros((len(values),3*len(channels)),np.float32);valid=np.zeros_like(y,dtype=bool)
    for i in range(len(values)):
        for j,ch in enumerate(channels):
            v=values[i,:,ch]; ok=status[i,:,ch]==0
            y[i,3*j]=v[-1] if ok[-1] else 0.;valid[i,3*j]=ok[-1]
            complete=True; up=False; down=False
            for t in range(1,len(v)):
                if not (ok[t-1] and ok[t]): complete=False;continue
                up=up or v[t]>v[t-1];down=down or v[t]<v[t-1]
            y[i,3*j+1:3*j+3]=[up,down]
            valid[i,3*j+1:3*j+3]=[up or complete,down or complete]
    return y,valid


def read_records(ds,records,channels,saved,kind):
    require(len(records)==1024,'Expected exactly 1024 '+kind+' windows')
    starts=[v['start'] for v in records]; w=ds.window_size
    require(np.all(np.diff(starts)>=w),'Overlapping or unordered '+kind+' windows')
    batches=[]; episodes=set()
    for pos in range(0,len(records),8):
        rec=records[pos:pos+8]
        b=window_batch(ds,[v['start'] for v in rec])
        for j,r in enumerate(rec):
            require(type(r['start']) is int and r['stop']-r['start']==w,'Invalid window bounds')
            require(int(b['source_start'][j])==r['source_start'] and int(b['source_stop'][j])==r['source_stop'],'Actual source rows differ')
            require(str(b['timestamp_end'][j])==r['timestamp_end'] and
                b['timestamp_end'][j]-np.timedelta64(w-1,'s')==np.datetime64(r['timestamp_start']),'Actual timestamps differ')
            require(np.all(b['status'][j]==0),'Selected window contains invalid input')
            if r['kind']=='event':
                require(r['episode_id'] is not None and r['episode_id'] not in episodes,'Repeated event episode')
                episodes.add(r['episode_id'])
            if r['kind']=='quiet':
                c=ds.read_rows(r['start']-w,r['stop']+w)
                require(np.all(c['status']==0),'Invalid quiet context')
                require(not np.any(np.diff(c['values'][:,channels],axis=0)),'Transition in quiet context')
        batches.append(b)
    b={k:np.concatenate([a[k] for a in batches]) for k in batches[0]}
    y,valid=targets(b['values'],b['status'],channels)
    oracle,oracle_valid=exact_rules(b['values'],b['status'],channels)
    require(np.array_equal(y,oracle) and np.array_equal(valid,oracle_valid),'Exact-rule target gate failed')
    for key,a in (('y',y),('valid',valid),('source_start',b['source_start']),
                  ('source_stop',b['source_stop']),('timestamp_end',b['timestamp_end'])):
        # Training selection archives store source coordinates in JSON records,
        # already checked above; their NPZ contains y/valid/starts only.
        if kind in ('update','fit') and key not in ('y','valid'): continue
        require(key in saved and np.array_equal(a,saved[key]),'Saved target/source differs: '+kind+'/'+key)
    b.update(y=y,valid=valid)
    return b


def prepare(args,out):
    sr=args.selection_report.resolve();er=args.evaluation_report.resolve()
    require(sha(sr.parent/'event_selection.json')==TRAIN_SELECTION,'Not the frozen training selection')
    require(sha(er.parent/'selection.json')==EVAL_SELECTION,'Not the frozen evaluation selection')
    ds={s:BREWindowDataset(args.input_dir,s,30) for s in ('train','actor1_test')}
    require(ds['train'].manifest_sha256==MANIFEST,'Prepared input manifest differs')
    m=read(sr.parent/'event_selection.json'); desc=m['targets']
    m,labels,deps=load_selection(sr,HISTORICAL_CHECKPOINT,ds['train'].metadata_hashes,
                               ORIGINAL_SELECTION,desc,30)
    require(ds['actor1_test'].metadata_hashes==ds['train'].metadata_hashes,'Split metadata differs')
    require(all(ds['train'].devices[d['channel_index']]==d['channel'] for d in desc),'Channel order differs')
    channels=[v['channel_index'] for v in desc[::3]]
    require(len(set(channels))==len(channels) and all(desc[3*j+k]['channel_index']==ch and
        desc[3*j+k]['kind']==('final_state','rise','fall')[k] for j,ch in enumerate(channels) for k in range(3)), 'Target layout differs')
    ev=read(er);require(ev['status']=='COMPLETE' and ev['protocol']=='bre_frozen_temporal_eval_v1','Wrong evaluation report')
    deps[str(er)]=sha(er)
    for name in ('selection.json','evaluation_contract.json','period_event_index.npz',
                 'uniform_targets.npz','event_enriched_targets.npz'):
        p=er.parent/name; require(name in ev['output_hashes'] and sha(p)==ev['output_hashes'][name],'Evaluation artifact differs: '+name)
        deps[str(p)]=sha(p)
    ec=read(er.parent/'evaluation_contract.json'); es=read(er.parent/'selection.json')
    active=np.flatnonzero(m['active_targets'])
    require(ec['targets']==desc and ec['active_targets']==active.tolist() and len(active)==12,'Active targets differ')
    require(ec['parent_checkpoint_sha256']==HISTORICAL_CHECKPOINT and es['split']=='actor1_test','Historical lineage differs')
    records={k:m['selections'][k]['windows'] for k in ('update','fit')}
    records.update(es['cohorts']); require(set(records)=={'update','fit','uniform','event_enriched'},'Unexpected cohorts')
    # Validate evaluation episode ownership and quiet contexts against frozen census.
    index=load_npz(er.parent/'period_event_index.npz'); rows=index['event_rows']; episodes=index['episode_index_ranges']
    for r in records['event_enriched']:
        if r['kind']=='event':
            e=r['episode_id'];require(type(e) is int and 0<=e<len(episodes),'Invalid evaluation episode')
            a,b=episodes[e];require(0<=a<b<=len(rows) and np.any((rows[a:b]>r['start'])&(rows[a:b]<r['stop'])),'Episode absent from window')
    write(out/'selections.json',dict(records=records,active_targets=active.tolist(),targets=desc,
        training_selection_sha256=TRAIN_SELECTION,evaluation_selection_sha256=EVAL_SELECTION))
    data={}
    for name in ('update','fit','uniform','event_enriched'):
        print('Verifying',name,'1024 windows',flush=True)
        saved=labels[name] if name in labels else load_npz(er.parent/(name+'_targets.npz'))
        data[name]=read_records(ds['train' if name in labels else 'actor1_test'],records[name],channels,saved,name)
    counts={name:support(b['y'],b['valid']) for name,b in data.items()}
    require(np.array_equal(eligible(counts['update'],counts['fit']),m['active_targets']),'Training support differs')
    require(data['update']['source_stop'].max()<=data['fit']['source_start'].min(),'Training source overlap')
    for cohort in ('uniform','event_enriched'):
        b=data[cohort]
        require(data['fit']['source_stop'].max()<=b['source_start'].min(),'Fit/evaluation source overlap')
        require(data['fit']['timestamp_end'].max()<b['timestamp_end'].min()-np.timedelta64(29,'s'),'Fit/evaluation temporal overlap')
        require(es['prior_source_stop']<=b['source_start'].min() and
                np.datetime64(es['prior_timestamp_end'])<b['timestamp_end'].min()-np.timedelta64(29,'s'),'Evaluation is not after prior development period')
    active_desc=[dict(target=int(j),**desc[j]) for j in active]
    primary=[i for i,d in enumerate(active_desc) if d['channel'] in ('OC-00-H1_contact_x','A6-01-H2_motion') and d['kind'] in ('rise','fall')]
    final=[i for i,d in enumerate(active_desc) if d['kind']=='final_state']
    require(len(primary)==4 and len(final)==6,'Fixed decision targets differ')
    for col in primary+final:
        j=active[col];require(min(counts['event_enriched']['positive'][j],counts['event_enriched']['negative'][j])>=20,'Primary evaluation coverage insufficient')
    for name,b in data.items():
        b['y']=b['y'][:,active];b['valid']=b['valid'][:,active]
        np.savez_compressed(out/(name+'_targets.npz'),**{k:b[k] for k in ('y','valid','source_start','source_stop','timestamp_end')})
    coverage=dict(targets=desc,active_targets=active.tolist(),counts=counts,
        primary_columns=primary,final_state_columns=final,
        selected_event_episodes={name:len({r['episode_id'] for r in rec if r['kind']=='event'}) for name,rec in records.items()},
        positive_event_episodes={name:[len({r['episode_id'] for i,r in enumerate(records[name])
            if r['kind']=='event' and b['valid'][i,j] and b['y'][i,j]==1}) for j in range(len(active))] for name,b in data.items()},
        uniform_overlapping_enriched=sum(any(a['start']<b['stop'] and b['start']<a['stop'] for b in records['event_enriched']) for a in records['uniform']))
    write(out/'coverage.json',coverage)
    return data,ds,deps,active_desc,primary,final,coverage
