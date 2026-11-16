from pathlib import Path
import csv,sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results/fluorescence_reconstruction'
sys.path.insert(0,str(ROOT/'src'))
from evaluation.spatial_response_data import read
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def val(r, k):
    return float(r[k])

def flag(r, k):
    return str(r[k]).lower() == 'true'

def grouped(rows, keys):
    d = defaultdict(list)
    for r in rows:
        d[tuple((r[k] for k in keys))].append(r)
    return d

def interval(rows, key, boolean=False):
    rows = [r for r in rows if boolean or np.isfinite(val(r, key))]
    if not rows:
        return (float('nan'), float('nan'), float('nan'))
    group = grouped(rows, ['group'])
    keys = sorted(group)
    cluster_means = {k: np.array([np.mean([float(flag(r, key)) if boolean else val(r, key) for r in q]) for q in grouped(values, ['well_id']).values()]) for k, values in group.items()}
    wellmeans = []
    for q in grouped(rows, ['well_id']).values():
        wellmeans.append(np.mean([float(flag(r, key)) if boolean else val(r, key) for r in q]))
    g = np.random.default_rng(203)
    boot = []
    for _ in range(1000):
        selected = np.concatenate([cluster_means[keys[i]] for i in g.integers(len(keys), size=len(keys))])
        boot.append(np.mean(selected))
    return (float(np.mean(wellmeans)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
def main():
 reconstruction=read(OUT/'RECONSTRUCTION_CELLS.csv');summary=[]
 for (assay,method,role),rows in grouped(reconstruction,['assay','method','role']).items():
  record=dict(assay=assay,method=method,role=role,wells=len({r['well_id'] for r in rows}),groups=len({r['group'] for r in rows}),cells=len(rows))
  for key in ['mae','foreground_mae','ssim']:
   p,l,h=interval(rows,key);record[key]=p;record[key+'_low']=l;record[key+'_high']=h
  eligible=[q for q in grouped(rows,['well_id']).values() if len({r['true_state'] for r in q})==2];record['marker_auroc']=float(np.mean([roc_auc_score([int(r['true_state']) for r in q],[val(r,'readout_pred') for r in q]) for q in eligible])) if eligible else float('nan');record['marker_ap']=float(np.mean([average_precision_score([int(r['true_state']) for r in q],[val(r,'readout_pred') for r in q]) for q in eligible])) if eligible else float('nan');record['discrimination_wells']=len(eligible);record['readout_mae']=float(np.mean([np.mean([abs(val(r,'readout_pred')-val(r,'readout_true')) for r in q]) for q in grouped(rows,['well_id']).values()]));summary.append(record)
 write('BASELINE_RESULTS.csv',summary)
 write('TNBC_RESULTS.csv',[r for r in summary if r['role'] in ['test','ood_test']])
if __name__=='__main__':main()
