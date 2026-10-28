"""Fit-only ridge readouts, update-only PCA, paired calendar-day uncertainty."""
import numpy as np
from src.scientific_validation.behavior_preservation.core import metrics


def pca_fit(x, width=320):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError('Invalid PCA input')
    mean=x.mean(0); _,s,vt=np.linalg.svd(x-mean,full_matrices=False)
    tolerance=max(x.shape)*np.finfo(np.float64).eps*(s[0] if len(s) else 0.)
    rank=int((s>tolerance).sum()); keep=min(width,rank)
    components=np.zeros((width,x.shape[1]),np.float64); components[:keep]=vt[:keep]
    # Canonicalize signs; repeated singular subspaces remain library-dependent.
    for row in components[:keep]:
        if row[np.argmax(np.abs(row))]<0: row *= -1
    return dict(mean=mean,components=components,singular_values=s,rank=rank,tolerance=tolerance)


def pca_apply(x,p):
    return (np.asarray(x,dtype=np.float64)-p['mean'])@p['components'].T


def fit_readouts(x,y,valid):
    """Same ridge=1 objective as fit_readout; share solves for equal validity masks."""
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or y.shape!=valid.shape or len(x)!=len(y) or not np.isfinite(x).all():
        raise ValueError('Invalid readout fit inputs')
    result={}; groups={}
    for j in range(y.shape[1]): groups.setdefault(valid[:,j].tobytes(),[]).append(j)
    for cols in groups.values():
        use=valid[:,cols[0]]
        if not use.any(): raise ValueError('No valid readout fit labels')
        a=x[use]; mean=a.mean(0); std=a.std(0); scale=np.where(std>1e-6,std,1.)
        z=(a-mean)/scale; truth=y[use][:,cols]; intercept=truth.mean(0); n,d=z.shape
        if d<=n: coef=np.linalg.solve(z.T@z+n*np.eye(d),z.T@(truth-intercept))
        else: coef=z.T@np.linalg.solve(z@z.T+n*np.eye(n),truth-intercept)
        for k,j in enumerate(cols):
            result[j]=dict(mean=mean,scale=scale,coef=coef[:,k],intercept=float(intercept[k]))
    return result


def predict(x,parameters):
    x=np.asarray(x,dtype=np.float64)
    scores=np.column_stack([((x-p['mean'])/p['scale'])@p['coef']+p['intercept']
                            for j,p in sorted(parameters.items())])
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite readout score')
    return scores


def per_target(y,valid,scores,descriptors):
    return [dict(**d,metrics=metrics(y[valid[:,j],j],scores[valid[:,j],j]),
        prevalence=float(y[valid[:,j],j].mean()) if valid[:,j].any() else None)
        for j,d in enumerate(descriptors)]


def macro(y,valid,scores,columns,key='average_precision'):
    vals=[metrics(y[valid[:,j],j],scores[valid[:,j],j])[key] for j in columns]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def day_bootstrap(y,valid,scores,times,columns,replicates=1000,seed=20261044):
    """scores: method -> seed -> N,T; all methods share each sampled set of days."""
    days=np.asarray(times).astype('datetime64[D]'); unique=np.unique(days)
    if len(unique)<2: raise ValueError('Temporal uncertainty requires at least two days')
    groups=[np.flatnonzero(days==d) for d in unique]; rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(unique),size=(replicates,len(unique)))
    values={name:np.full(replicates,np.nan) for name in scores}
    for b,draw in enumerate(draws):
        ix=np.concatenate([groups[i] for i in draw])
        if any(len(np.unique(y[ix,j][valid[ix,j]]))<2 for j in columns): continue
        for name,runs in scores.items():
            values[name][b]=np.mean([macro(y[ix],valid[ix],s[ix],columns) for s in runs])
    differences={name:values['trained_dbrl']-values[name]
                 for name in ('trained_temporal','initial_dbrl')}
    intervals={}
    for name,a in differences.items():
        good=np.isfinite(a); reliable=bool(good.mean()>=.95)
        intervals[name]=dict(lower=float(np.quantile(a[good],.025)) if good.any() else None,
            upper=float(np.quantile(a[good],.975)) if good.any() else None,
            undefined=int((~good).sum()),replicates=replicates,reliable=reliable)
    return dict(days=unique.astype(str).tolist(),intervals=intervals),dict(draws=draws,**differences)


def decision(native,compact,final_state,uncertainty,resources_reported):
    d=np.array(native['trained_dbrl']); checks={}
    for name in ('trained_temporal','initial_dbrl'):
        diff=d-np.array(native[name]); ci=uncertainty['intervals'][name]
        checks['native_'+name]=bool(diff.mean()>=.03 and np.all(diff>0))
        checks['interval_'+name]=bool(ci['reliable'] and ci['lower'] is not None and ci['lower']>0)
    for name in ('trained_temporal','raw'):
        checks['compact_'+name]=bool(np.mean(compact['trained_dbrl'])-np.mean(compact[name])>=.03)
    checks['final_state_noninferiority']=bool(np.mean(final_state['trained_dbrl'])-np.mean(final_state['trained_temporal'])>=-.02)
    checks['resources_reported']=bool(resources_reported)
    unreliable=any(not v['reliable'] for v in uncertainty['intervals'].values())
    outcome='INCONCLUSIVE' if unreliable else ('CONTINUE_DBRL_VALIDATION' if all(checks.values()) else 'STOP_GRAPH_DEVELOPMENT')
    return dict(outcome=outcome,checks=checks,
        scope='Development decision only; no anomaly, human-activity or publication claim.')


def summaries(values,status):
    ok=status==0; count=ok.sum(1); any_valid=count>0
    mean=np.where(ok,values,0.).sum(1)/np.maximum(count,1)
    var=np.where(ok,(values-mean[:,None])**2,0.).sum(1)/np.maximum(count,1)
    last_index=values.shape[1]-1-np.argmax(ok[:,::-1],axis=1)
    last=np.take_along_axis(values,last_index[:,None,:],axis=1)[:,0]
    last=np.where(any_valid,last,0.)
    endpoints=ok[:,0]&ok[:,-1]; increment=np.where(endpoints,values[:,-1]-values[:,0],0.)
    return np.concatenate([last,mean,np.sqrt(var),increment,any_valid.astype(float),
                           endpoints.astype(float),count/values.shape[1]],1)
