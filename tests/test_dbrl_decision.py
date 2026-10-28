import copy
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits
from src.scientific_validation.dbrl_decision.model import (
    paired_models,state_hash,schedule,backward_gate,train_arm,extract)
from src.scientific_validation.dbrl_decision.statistics import (
    pca_fit,pca_apply,fit_readouts,predict,day_bootstrap,decision,summaries,macro)
from src.scientific_validation.dbrl_decision.inputs import exact_rules,read_records
from src.scientific_validation.behavior_preservation.core import targets
from run_v4_bre_representation_probe import fit_readout,predict_readout


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): yield


def batch(n=16,w=30,c=3):
    rng=np.random.default_rng(12)
    values=rng.integers(0,2,(n,w,c)).astype(np.float32)
    status=np.zeros_like(values,dtype=np.uint8)
    return dict(values=values,status=status,
                X=np.concatenate([values[...,None],np.eye(5,dtype=np.float32)[status]],-1))


def test_pair_exact_temporal_and_decoder_initialization():
    a=paired_models(43,3);b=batch()
    x=torch.from_numpy(b['X'])
    assert torch.equal(a['temporal'].encoder(x),a['dbrl'].encoder._encode_temporal_behavior(x))
    assert state_hash(a['temporal'].decoder)==state_hash(a['dbrl'].decoder)
    assert not any('graph' in k for k in a['temporal'].state_dict())
    assert state_hash(a['dbrl'])==state_hash(paired_models(43,3)['dbrl'])


def test_fixed_schedule_eight_complete_passes():
    a=schedule(20261041)
    assert a.shape==(1024,8) and np.array_equal(a,schedule(20261041))
    for epoch in a.reshape(8,1024): assert np.array_equal(np.sort(epoch),np.arange(1024))


@pytest.mark.parametrize('arm',['temporal','dbrl'])
def test_backward_and_bounded_optimizer_integration(arm):
    model=paired_models(42,3)[arm];b=batch();dev=torch.device('cpu')
    old=state_hash(model);backward_gate(model,b,dev);assert state_hash(model)==old
    result=train_arm(model,b,np.arange(16).reshape(2,8),dev)
    assert result['steps']==2 and result['parameter_delta_l2']['encoder']>0
    assert state_hash(model)!=old
    h=state_hash(model);z,e,t=extract(model,b,dev)
    assert z.shape==(16,192) and np.isfinite(e).all() and t>0
    assert h==state_hash(model)


def test_ridge_batched_matches_existing_individual_validity_masks():
    rng=np.random.default_rng(6);x=rng.normal(size=(25,40));test=rng.normal(size=(11,40))
    y=rng.integers(0,2,(25,3)).astype(float);valid=np.ones_like(y,dtype=bool);valid[:4,2]=False
    params=fit_readouts(x,y,valid);out=predict(test,params)
    for j in range(3):
        old=fit_readout(x[valid[:,j]],y[valid[:,j],j])
        np.testing.assert_allclose(out[:,j],predict_readout(test,old),rtol=1e-10,atol=1e-10)


def test_pca_update_only_rank_padding_and_replay():
    x=np.arange(48,dtype=float).reshape(12,4)
    p=pca_fit(x,width=8);mean=p['mean'].copy();q=pca_apply(x+1000,p)
    assert p['rank']==1 and q.shape==(12,8)
    assert np.array_equal(mean,p['mean']) and np.all(q[:,1:]==0)
    np.testing.assert_allclose(pca_apply(x,p)[:,0].mean(),0,atol=1e-10)


def test_exact_rules_handle_missing_and_observed_positive():
    b=batch(n=5);b['status'][0,5,0]=3;b['status'][1,-1,1]=2
    a=targets(b['values'],b['status'],[0,1,2]);z=exact_rules(b['values'],b['status'],[0,1,2])
    assert all(np.array_equal(x,y) for x,y in zip(a,z))
    assert not z[1][1,3]


def test_summary_missing_indicators_and_finiteness():
    b=batch(n=2);b['status'][0,:,0]=3
    s=summaries(b['values'],b['status'])
    assert s.shape==(2,21) and np.isfinite(s).all()
    assert np.all(s[0,[0,3,6,9,12,15,18]]==0)


def test_day_bootstrap_paired_seeds_not_pooled():
    y=np.tile([[0.],[1.]],(10,1));valid=np.ones_like(y,dtype=bool)
    days=np.repeat(np.arange('2022-01-01','2022-01-06',dtype='datetime64[D]'),4)
    a=y*.9;bad=1-a
    scores=dict(trained_dbrl=[a,bad],trained_temporal=[bad,a],initial_dbrl=[a,a])
    u,d=day_bootstrap(y,valid,scores,days,[0],replicates=30)
    np.testing.assert_allclose(d['trained_temporal'],0)
    assert u['intervals']['trained_temporal']['reliable'] and d['draws'].shape==(30,5)


def test_undefined_days_produce_inconclusive():
    y=np.array([[0.],[0.],[1.],[1.]]);v=np.ones_like(y,dtype=bool)
    days=np.array(['2022-01-01']*2+['2022-01-02']*2,dtype='datetime64[D]')
    scores={k:[y] for k in ('trained_dbrl','trained_temporal','initial_dbrl')}
    u,d=day_bootstrap(y,v,scores,days,[0],replicates=100)
    assert not u['intervals']['trained_temporal']['reliable']
    n={k:[.8]*3 for k in scores};c=dict(trained_dbrl=[.8]*3,trained_temporal=[.5]*3,raw=[.4])
    assert decision(n,c,n,u,True)['outcome']=='INCONCLUSIVE'


def test_stop_rule_requires_every_condition():
    n=dict(trained_dbrl=[.8]*3,trained_temporal=[.7]*3,initial_dbrl=[.65]*3)
    c=dict(trained_dbrl=[.8]*3,trained_temporal=[.7]*3,raw=[.7])
    u=dict(intervals={k:dict(lower=.01,reliable=True) for k in ('trained_temporal','initial_dbrl')})
    assert decision(n,c,n,u,True)['outcome']=='CONTINUE_DBRL_VALIDATION'
    bad=copy.deepcopy(n);bad['trained_dbrl'][0]=.6
    assert decision(bad,c,n,u,True)['outcome']=='STOP_GRAPH_DEVELOPMENT'
    assert decision(n,c,n,u,False)['outcome']=='STOP_GRAPH_DEVELOPMENT'


def test_invalid_input_bounds_fail_before_loader_reads():
    class DS: window_size=30
    with pytest.raises(ValueError,match='1024'):
        read_records(DS(),[],[0],{},'test')


def test_undefined_macro_not_silently_dropped():
    y=np.zeros((4,2));y[:,0]=[0,1,0,1]
    assert macro(y,np.ones_like(y,dtype=bool),y,[0,1]) is None


def test_saved_readout_pipeline_replays_and_never_fits_probe(tmp_path):
    from run_v4_bre_dbrl_decision import evaluate_features
    rng=np.random.default_rng(52);features={};data={}
    for name in ('update','fit','uniform','event_enriched'):
        f=rng.normal(size=(24,10));y=(f[:,:3]>0).astype(float)
        features[name]=f
        data[name]=dict(y=y,valid=np.ones_like(y,dtype=bool),source_start=np.arange(24),
            source_stop=np.arange(24)+30,timestamp_end=np.arange(24).astype('datetime64[s]'))
    desc=[dict(target=i,kind='rise' if i<2 else 'final_state') for i in range(3)]
    report={'readouts':{}};bank={}
    evaluate_features('example',features,data,desc,[0,1],[2],tmp_path,report,bank)
    saved=np.load(tmp_path/'example_native_readouts.npz')
    params={i:{k:saved[str(i)+'_'+k] for k in ('mean','scale','coef','intercept')} for i in range(3)}
    scores=np.load(tmp_path/'example_native_event_enriched_scores.npz')['scores']
    np.testing.assert_allclose(scores,predict(features['event_enriched'],params))
    p=dict(np.load(tmp_path/'example_pca.npz'))
    np.testing.assert_allclose(p['mean'],features['update'].mean(0))
    assert p['components'].shape==(320,10) and len(bank)==2
    # A changed evaluation label cannot affect fitting parameters.
    copy_data=copy.deepcopy(data);copy_data['event_enriched']['y']=1-copy_data['event_enriched']['y']
    next_dir=tmp_path/'other';next_dir.mkdir()
    evaluate_features('example',features,copy_data,desc,[0,1],[2],next_dir,{'readouts':{}},{})
    with np.load(next_dir/'example_native_readouts.npz') as other:
        for k in saved.files: np.testing.assert_array_equal(saved[k],other[k])
