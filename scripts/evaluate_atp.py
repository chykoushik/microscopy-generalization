from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluation.atp_risk import joint_source_selection
def main():
 source=ROOT/'outputs/results/atp_source_selection'
 previous=json.loads((ROOT/'outputs/results/atp_risk_baselines/results.json').read_text())
 recorded=json.loads((source/'results.json').read_text())
 for path in sorted(source.glob('A_raw_*.json')):
  fold=json.loads(path.read_text());study=fold['study'];ids=fold['well_ids'];matrix=np.asarray(fold['R'])
  rows={r['index']:r for r in previous['records'] if r['setting']=='A_raw' and r['study']==study}
  errors=np.array([rows[i]['absolute_error'] for i in fold['test_indices']])
  for key,methods in fold['selections'].items():
   coverage=float(key)
   solved=joint_source_selection(matrix,ids,int(np.ceil(coverage*len(ids))))
   assert solved['retained_indices']==methods['joint_SRSO']
   for method,keep in methods.items():
    expected=next(r for r in recorded['metrics'] if r['study']==study and r['method']==method and r['coverage']==coverage)
    np.testing.assert_allclose(errors[keep].mean(),expected['mae_nM'],rtol=0,atol=1e-10)
 print('ATP replay passed')
if __name__=='__main__':main()
