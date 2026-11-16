import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluation.image_editing_data import measured_state
from evaluation.image_editing_models import SmallUNet,inputs,predict
OUT=ROOT/'outputs/results/fluorescence_reconstruction'
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_model(assay,kind,seed):
 p=torch.load(OUT/(assay+'_'+kind+'_'+str(seed)+'.pt'),weights_only=False)
 m=SmallUNet(p['inputs'],p['outputs']);m.load_state_dict(p['state_dict']);return m.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cfg=json.loads((OUT/'CONFIG.json').read_text());reconstruction=[]
 for config in cfg['assays']:
  assay=config['assay'];data=np.load(OUT/(assay+'_images.npz'));images=data['images'].astype(np.float32);n=data['nuclei'];c=data['cytoplasm'];cell=data['cells'];states=data['state'].astype(int);desc=data['descriptor'];meta=json.loads((OUT/(assay+'_cells.json')).read_text());train=np.array([i for i,w in enumerate(meta) if w['role']=='train']);query=np.array([i for i,w in enumerate(meta) if w['role']!='train']);scale=StandardScaler().fit(desc[train]);z=scale.transform(desc);distance=cdist(z[query],z[train]);nearest=train[distance.argmin(1)];recon={};full=[]
  for seed in [200,201]:full.append(predict(load_model(assay,'reconstruction',seed),inputs(images,n,c,'reconstruction'))[:,0])
  recon['unet']=np.mean(full,0)[query];recon['zero']=np.zeros_like(recon['unet']);recon['copy_structural']=images[query,2];recon['nearest_structural']=images[nearest,1]
  for kind in ['dapi','structural']:recon['unet_'+kind]=predict(load_model(assay,kind,200),inputs(images,n,c,kind))[query,0]
  np.savez_compressed(OUT/(assay+'_RECONSTRUCTIONS.npz'),indices=query,original=images[query].astype(np.float16),**{k:v.astype(np.float16) for k,v in recon.items()})
 write('RECONSTRUCTION_CELLS.csv',reconstruction)
if __name__=='__main__':main()
