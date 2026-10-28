"""Read-only verification of the frozen event-aware selection."""
import json
from pathlib import Path
import numpy as np
from src.scientific_validation.reconstruction.runtime import require,sha


def load_selection(path,checkpoint_hash,input_hashes,original_hash,descriptors,window):
    path=Path(path);root=path.parent;r=json.loads(path.read_text())
    require(r['protocol']=='bre_train_event_selection_v1' and r['status']=='COMPLETE' and r['support_gate_pass'],'Selection gate did not pass')
    deps={str(path):sha(path)}
    for name,h in r['output_hashes'].items():
        require(Path(name).name==name and sha(root/name)==h,'Selection artifact differs: '+name)
        deps[str(root/name)]=h
    m=json.loads((root/'event_selection.json').read_text());c=json.loads((root/'selection_contract.json').read_text())
    require(m['usable_for_behavior_experiment'] and m['protocol']=='bre_train_event_selection_v1','Unusable manifest')
    require(m['contract_sha256']==sha(root/'selection_contract.json'),'Contract differs')
    require(m['parent_checkpoint_sha256']==c['parent_checkpoint_sha256']==checkpoint_hash,'Checkpoint differs')
    require(m['input_metadata_hashes']==input_hashes and m['original_selection_sha256']==original_hash,'Input or original selection differs')
    require(m['targets']==c['targets']==descriptors and m['window_size']==c['window_size']==window,'Target contract differs')
    require(not m['heldout_values_read'] and not r['heldout_values_read'],'Selection read held-out values')
    with np.load(root/'training_event_index.npz',allow_pickle=False) as a:index={key:a[key] for key in a.files}
    verify_geometry(m,c,index)
    labels={}
    for name in ('update','fit'):
        with np.load(root/(name+'_targets.npz'),allow_pickle=False) as a:labels[name]={key:a[key] for key in a.files}
        require(np.array_equal(labels[name]['starts'],[v['start'] for v in m['selections'][name]['windows']]),'Target starts differ')
    return m,labels,deps


def verify_geometry(m,c,index):
    w=m['window_size'];regions=c['regions'];boundary=c['selection_boundary']
    require(regions['update'][1]==boundary-w and regions['fit'][0]==boundary+w,'Guard differs')
    rows=index['event_rows'];episodes=index['episode_index_ranges'];bad=index['invalid_rows'];seen=set();sources=[]
    require(np.all(np.diff(rows)>0) and np.all(np.diff(bad)>0),'Index order differs')
    for name in ('update','fit'):
        part=m['selections'][name];lo,hi=regions[name];last=lo;records=part['windows']
        require(part['split']=='train' and part['region']==regions[name],'Region differs')
        require(len(records)==1024 and sum(v['kind']=='quiet' for v in records)>=256,'Budget differs')
        for v in records:
            start=v['start'];stop=v['stop'];require(lo<=start<stop<=hi and stop-start==w and start>=last,'Overlapping or invalid interval');last=stop
            require(v['source_stop']-v['source_start']==w,'Source length differs');sources.append((v['source_start'],v['source_stop']))
            require(np.datetime64(v['timestamp_end'])-np.datetime64(v['timestamp_start'])==np.timedelta64(w-1,'s'),'Clock differs')
            if v['kind']=='event':
                e=v['episode_id'];require(type(e) is int and 0<=e<len(episodes) and e not in seen,'Repeated or invalid episode');seen.add(e)
                a,b=episodes[e];require(0<=a<b<=len(rows) and lo<=rows[a] and rows[b-1]<hi,'Episode crosses region')
                require(np.any((rows[a:b]>start)&(rows[a:b]<stop)),'Assigned episode absent from window')
            elif v['kind']=='quiet':
                require(v['episode_id'] is None and lo<=start-w and stop+w<=hi,'Quiet context crosses region')
                require(np.searchsorted(rows,stop+w)==np.searchsorted(rows,start-w),'Quiet context contains event')
                require(np.searchsorted(bad,stop+w)==np.searchsorted(bad,start-w),'Quiet context contains invalid row')
            else:raise ValueError('Unknown window kind')
    sources.sort();require(all(a[1]<=b[0] for a,b in zip(sources,sources[1:])),'Source overlap')
    require(m['selections']['update']['windows'][-1]['source_stop']<=m['selections']['fit']['windows'][0]['source_start'],'Source chronology differs')
