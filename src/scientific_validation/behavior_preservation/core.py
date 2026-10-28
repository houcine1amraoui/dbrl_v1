"""Observable binary-state targets and fixed behavior-preservation utilities."""
import numpy as np
import torch
from torch import nn


def targets(values,status,channels):
    x=np.asarray(values)[:,:,channels];ok=np.asarray(status)[:,:,channels]==0
    if not np.all(np.isin(x[ok],[0.,1.])):
        raise ValueError('Valid behavioral inputs must be binary')
    paired=ok[:,1:]&ok[:,:-1]
    rise=(paired&(x[:,1:]>x[:,:-1])).any(1)
    fall=(paired&(x[:,1:]<x[:,:-1])).any(1)
    complete=paired.all(1)
    y=np.stack([np.where(ok[:,-1],x[:,-1],0),rise,fall],-1).astype(np.float32)
    valid=np.stack([ok[:,-1],rise|complete,fall|complete],-1)
    return y.reshape(len(x),-1),valid.reshape(len(x),-1)


def support(y,valid):
    return dict(positive=((y==1)&valid).sum(0).tolist(),negative=((y==0)&valid).sum(0).tolist(),
                missing=(~valid).sum(0).tolist())


def eligible(update,fit,floor=20):
    return np.array([(min(update['positive'][i],update['negative'][i],fit['positive'][i],fit['negative'][i])>=floor)
                     for i in range(len(update['positive']))])


class BehavioralHeads(nn.Module):
    def __init__(self,dim,state_indices):
        super().__init__()
        self.register_buffer('state_indices',torch.tensor(state_indices,dtype=torch.long))
        self.weight=nn.Parameter(torch.zeros(len(state_indices),dim))
        self.bias=nn.Parameter(torch.zeros(len(state_indices)))

    def forward(self,s):
        return (s[:,self.state_indices]*self.weight).sum(-1)+self.bias


def behavioral_loss(logits,y,valid,pos_weight,active):
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight,reduction='none')
    counts=valid.sum(0);use=active&(counts>0)
    if not use.any():return logits.sum()*0
    per_target=torch.where(valid,losses,torch.zeros_like(losses)).sum(0)/counts.clamp_min(1)
    return per_target[use].mean()


def metrics(y,scores):
    """Fixed 0.5 threshold; AP groups equal scores (no favorable tie ordering)."""
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float)
    if not len(y):return dict(n=0,positive=0,negative=0,balanced_accuracy=None,precision=None,recall=None,average_precision=None)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite readout scores')
    p=y==1;n=y==0;pred=scores>=.5;tp=int((p&pred).sum());fp=int((n&pred).sum())
    recall=tp/int(p.sum()) if p.any() else None
    ba=(recall+float((n&~pred).sum()/n.sum()))/2 if p.any() and n.any() else None
    ap=None
    if p.any() and n.any():
        order=np.argsort(-scores,kind='stable');ys=y[order];ss=scores[order]
        ends=np.r_[np.where(ss[:-1]!=ss[1:])[0],len(ss)-1]
        tp_curve=np.cumsum(ys)[ends];rec=tp_curve/p.sum();prec=tp_curve/(ends+1)
        ap=float(np.sum(np.diff(np.r_[0.,rec])*prec))
    return dict(n=len(y),positive=int(p.sum()),negative=int(n.sum()),balanced_accuracy=ba,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=recall,average_precision=ap)


def reconstruction_loss(pred,values,status):
    valid=status==0;counts=valid.sum((1,2))
    if not bool((counts>0).all()):raise ValueError('No valid reconstruction targets')
    return (torch.where(valid,(pred-values).square(),torch.zeros_like(pred)).sum((1,2))/counts).mean()
