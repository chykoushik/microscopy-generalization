import os
os.environ['OMP_NUM_THREADS']='1'
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results/corruption_robustness'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'outputs/results/fluorescence_reconstruction'
sys.path.insert(0,str(ROOT/'src'))
from evaluation.image_editing_data import descriptor
SEEDS=[719,1729,2719]
CONDITIONS=['clean']+[f'{kind}_{level}' for kind in ['blur','dropout','saturation','occlusion','boundary'] for level in [1,2]]
NAMES=['RF','boosting','RF_aug','boosting_aug','uniform','attention','quality','uniform_aug','attention_aug','RGME_no_stability','RGME']


def save(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(y,p):
    bins=np.minimum((p*10).astype(int),9)
    return dict(ap=float(average_precision_score(y,p)) if len(set(y))==2 else None,auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,brier=float(np.mean((y-p)**2)),ece=float(sum(np.mean(bins==b)*abs(y[bins==b].mean()-p[bins==b].mean()) for b in set(bins))),prevalence=float(y.mean()),cells=len(y))


def features(images,n,c):
    d=descriptor(images,n,c)
    x=np.stack([np.c_[d[:,:4],d[:,4:14]],np.c_[d[:,:4],d[:,14:24]]],1)
    quality=[]
    for channel in [0,2]:
        a=images[:,channel];lap=np.stack([ndi.laplace(z) for z in a])
        quality.append(np.c_[np.log1p(lap.var((1,2))*100), (a>=.99).mean((1,2)),a.std((1,2))])
    return x.astype(np.float32),np.stack(quality,1).astype(np.float32)


def corrupt(images,n,c,kind,level,which):
    a=images.copy();nn=n.copy();cc=c.copy()
    for i,k in enumerate(which):
        ch=0 if k==0 else 2
        if kind=='blur':a[i,ch]=ndi.gaussian_filter(a[i,ch],1 if level==1 else 3)
        elif kind=='dropout':
            if level==1:a[i,ch,:,::2]=0
            else:a[i,ch]=0
        elif kind=='saturation':a[i,ch]=np.clip(a[i,ch]*(2 if level==1 else 5),0,1)
        elif kind=='occlusion':
            width=16 if level==1 else 32;a[i,ch,16:16+width,16:16+width]=0
        elif kind=='boundary':
            # Shared mask perturbation.
            nn[i]=ndi.binary_dilation(n[i],iterations=1 if level==1 else 3)
            cc[i]=c[i]&~nn[i]
            if cc[i].sum()<5:cc[i]=c[i]
    return a,nn,cc


class Model(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.enc=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(14,16),torch.nn.ReLU()) for _ in range(2)])
        self.gate=torch.nn.Sequential(torch.nn.Linear(19,12),torch.nn.Tanh(),torch.nn.Linear(12,1))
        self.head=torch.nn.Linear(16,1)

    def forward(self,x,q):
        z=torch.stack([e(x[:,i]) for i,e in enumerate(self.enc)],1)
        raw=self.gate(torch.cat([z,q],-1)).squeeze(-1)
        if self.kind.startswith('uniform'):a=torch.ones_like(raw)/2
        elif self.kind=='quality':
            r=torch.sigmoid(q[:,:,0]-q[:,:,1]+q[:,:,2]);a=r/(r.sum(1,keepdim=True)+1e-6)
        elif self.kind.startswith('RGME'):
            r=torch.sigmoid(raw);a=r/(r.sum(1,keepdim=True)+1e-6)
        else:a=torch.softmax(raw,1)
        return self.head((a[:,:,None]*z).sum(1)).squeeze(-1),a


def main():
    started=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=dict(seeds=SEEDS,conditions=CONDITIONS,epochs=30,lambda_stability=.5,learning_rate=.003,hidden=16,primary='mean corrupted macro-well AP; clean harm separately',device='CPU: tiny fixed-feature heads; no encoder',selection='validation mean across predefined corruptions; no test selection')
    (OUT/'CONFIG.json').write_text(json.dumps(config,indent=2))
    path=SOURCE/'GAMMA_images.npz';data=np.load(path);images=data['images'].astype(np.float32);n=data['nuclei'];c=data['cytoplasm'];y=data['state'].astype(int);meta=json.loads((SOURCE/'GAMMA_cells.json').read_text())
    assert len(images)==len(meta)==1646 and len({r['id'] for r in meta})==1646
    readout=(images[:,1]*n).sum((1,2))/n.sum((1,2));assert np.max(abs(readout-data['readout']))<1e-4
    roles=np.array([r['role'] for r in meta]);wells=np.array([r['well_id'] for r in meta]);tr=np.where(roles=='train')[0];va=np.where(roles=='validation')[0]
    for role in set(roles):
        for other in set(roles):
            if role!=other:assert not set(wells[roles==role])&set(wells[roles==other])
    clean,qclean=features(images,n,c)
    center=clean[tr].mean(0);scale=np.maximum(clean[tr].std(0),1e-3);qc=qclean[tr].mean(0);qs=np.maximum(qclean[tr].std(0),1e-3)
    counts=np.array([sum(wells[tr]==wells[i]) for i in tr]);weights=1/counts;weights/=weights.mean()
    results=[];predictions=[];gates=[];counter=[];runtime=[];histories=[]
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);which=rng.integers(0,2,len(y));cache={}
        for condition in CONDITIONS:
            if condition=='clean':x,q=clean,qclean
            else:
                kind,level=condition.rsplit('_',1);a,nn,cc=corrupt(images,n,c,kind,int(level),which);x,q=features(a,nn,cc)
            cache[condition]=(np.clip((x-center)/scale,-12,12).astype(np.float32),np.clip((q-qc)/qs,-12,12).astype(np.float32))
        x0,q0=cache['clean'];aug=np.array([rng.integers(1,len(CONDITIONS)) for _ in tr]);xa=np.stack([cache[CONDITIONS[j]][0][i] for i,j in zip(tr,aug)]);qa=np.stack([cache[CONDITIONS[j]][1][i] for i,j in zip(tr,aug)])
        flat=lambda x,q:np.concatenate([x.reshape(len(x),-1),q.reshape(len(q),-1)],1)
        for name in NAMES:
            t=time.perf_counter();neural=name not in ['RF','boosting','RF_aug','boosting_aug']
            if not neural:
                trainx=flat(x0[tr],q0[tr]);target=y[tr];ww=weights
                if name.endswith('_aug'):trainx=np.r_[trainx,flat(xa,qa)];target=np.tile(target,2);ww=np.tile(ww,2)
                model=RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=3,random_state=seed,n_jobs=1) if name.startswith('RF') else HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=1,random_state=seed)
                model.fit(trainx,target,sample_weight=ww)
                def predict(x,q):return model.predict_proba(flat(x,q))[:,1],np.ones((len(x),2))*.5
            else:
                torch.manual_seed(seed);model=Model(name);opt=torch.optim.Adam(model.parameters(),lr=.003)
                tx=torch.tensor(x0[tr]);tq=torch.tensor(q0[tr]);ax=torch.tensor(xa);aq=torch.tensor(qa);ty=torch.tensor(y[tr],dtype=torch.float32);tw=torch.tensor(weights,dtype=torch.float32)
                augmented=name.endswith('_aug') or name.startswith('RGME')
                for epoch in range(30):
                    opt.zero_grad();logit,a=model(tx,tq);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,ty,reduction='none')*tw).mean()
                    stability=torch.tensor(0.)
                    if augmented:
                        alt,_=model(ax,aq);loss=.5*(loss+(torch.nn.functional.binary_cross_entropy_with_logits(alt,ty,reduction='none')*tw).mean())
                        if name=='RGME':stability=((torch.sigmoid(logit)-torch.sigmoid(alt))**2*tw).mean();loss+=.5*stability
                    loss.backward();opt.step()
                histories.append(dict(seed=seed,method=name,loss=float(loss.detach()),stability=float(stability.detach()),parameters=sum(p.numel() for p in model.parameters())))
                model.eval()
                def predict(x,q):
                    with torch.no_grad():z,a=model(torch.tensor(x),torch.tensor(q));return torch.sigmoid(z).numpy(),a.numpy()
            runtime.append(dict(seed=seed,method=name,train_seconds=time.perf_counter()-t))
            pc,ac=predict(x0,q0)
            for condition,(x,q) in cache.items():
                pp,aa=predict(x,q)
                for role in ['validation','test','ood_test']:
                    ids=np.where(roles==role)[0];per=[]
                    for well in sorted(set(wells[ids])):
                        ii=ids[wells[ids]==well];m=metrics(y[ii],pp[ii]);per.append(m)
                        predictions.append(dict(seed=seed,method=name,condition=condition,role=role,well=well,**m,stability=float(np.mean(abs(pp[ii]-pc[ii])))))
                    means={k:float(np.mean([r[k] for r in per if r[k] is not None])) for k in ['ap','auroc','brier','ece']}
                    results.append(dict(seed=seed,method=name,condition=condition,role=role,cells=len(ids),wells=len(per),prevalence=float(y[ids].mean()),**means))
                    if neural and condition!='clean':
                        if condition.startswith('boundary'):cleanweight=degradedweight=None
                        else:degradedweight=float(aa[ids,which[ids]].mean());cleanweight=float(aa[ids,1-which[ids]].mean())
                        gates.append(dict(seed=seed,method=name,condition=condition,role=role,degraded_weight=degradedweight,clean_weight=cleanweight,wells=len(per)))
            if neural:
                for role in ['test','ood_test']:
                    ids=np.where(roles==role)[0]
                    for rank,unit in [('highest',ac.argmax(1)),('lowest',ac.argmin(1))]:
                        im,nn,cc=corrupt(images,n,c,'blur',2,unit);fx,fq=features(im,nn,cc);fx=np.clip((fx-center)/scale,-12,12).astype(np.float32);fq=np.clip((fq-qc)/qs,-12,12).astype(np.float32);pp,_=predict(fx,fq)
                        counter.append(dict(seed=seed,method=name,role=role,rank=rank,mean_absolute_probability_change=float(np.mean(abs(pp[ids]-pc[ids]))),wells=len(set(wells[ids])),cells=len(ids)))
            print('Model completed',flush=True)
    save('03_BASELINE_RESULTS.csv',[r for r in results if not r['method'].startswith('RGME')]);save('05_RGME_RESULTS.csv',[r for r in results if r['method'].startswith('RGME')]);save('PER_WELL.csv',predictions);save('GATE_WEIGHTS.csv',gates);save('COUNTERFACTUAL.csv',counter);save('RUNTIME.csv',runtime);save('TRAINING.csv',histories)
    summary=[]
    for role in ['test','ood_test']:
        for name in NAMES:
            own=[r for r in results if r['role']==role and r['method']==name];cleanrows=[r for r in own if r['condition']=='clean'];bad=[r for r in own if r['condition']!='clean'];summary.append(dict(role=role,method=name,clean_ap=float(np.mean([r['ap'] for r in cleanrows])),corrupted_ap=float(np.mean([r['ap'] for r in bad])),worst_ap=min(float(np.mean([r['ap'] for r in bad if r['condition']==condition])) for condition in CONDITIONS[1:]),clean_brier=float(np.mean([r['brier'] for r in cleanrows])),corrupted_brier=float(np.mean([r['brier'] for r in bad]))))
    save('SUMMARY.csv',summary)
    paired=[]
    for role in ['test','ood_test']:
        for condition in ['clean','corrupted']:
            valid=lambda r:r['condition']=='clean' if condition=='clean' else r['condition']!='clean'
            for other in [n for n in NAMES if n!='RGME']:
                d=[]
                for well in sorted(set(wells[roles==role])):
                    a=[r['ap'] for r in predictions if r['role']==role and r['well']==well and r['method']=='RGME' and valid(r) and r['ap'] is not None];b=[r['ap'] for r in predictions if r['role']==role and r['well']==well and r['method']==other and valid(r) and r['ap'] is not None]
                    if a and b:d.append(np.mean(a)-np.mean(b))
                rng=np.random.default_rng(719);boot=rng.choice(d,(500,len(d))).mean(1)
                paired.append(dict(role=role,condition=condition,comparison='RGME minus '+other,ap_delta=float(np.mean(d)),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),wells=len(d)))
    save('PAIRED.csv',paired)
    selected=max([name for name in NAMES if name!='RGME'],key=lambda name:np.mean([r['ap'] for r in results if r['role']=='validation' and r['method']==name and r['condition']!='clean']))
    count=[dict(role=role,cells=int((roles==role).sum()),wells=len(set(wells[roles==role])),positives=int(y[roles==role].sum())) for role in sorted(set(roles))];save('COUNTS.csv',count)
    manifest=dict(cells=len(y),wells=len(set(wells)),images=len(y),channels=3,permitted_channels=['DAPI','Cy5_tubulin'],target='saved nuclear Cy3 intensity state; not spot-count endpoint',source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((SOURCE/'GAMMA_cells.json').read_bytes()).hexdigest(),selected_baseline=selected,config=config,runtime_seconds=time.perf_counter()-started,python=platform.python_version(),torch=torch.__version__,cuda_used=False,downloads=0,E_access=False,feature_family='fixed14D same-cell channel descriptors plus3quality coordinates; no learned encoder',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'11_REPRODUCIBILITY_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('Experiment completed',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
