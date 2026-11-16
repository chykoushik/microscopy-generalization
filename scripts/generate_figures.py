import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path
import urllib.request
import zipfile

import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results/manuscript_figures'
OUT.mkdir(parents=True,exist_ok=True)
MM=ROOT/'outputs/results/missing_marker'
EDIT=ROOT/'outputs/results/fluorescence_reconstruction'
JUMP=ROOT/'outputs/results/jump_minority_signal_v1'
RGME=ROOT/'outputs/results/corruption_robustness'
for directory in ['figures','tables','qa']: (OUT/directory).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
evidence=[];hashes={};captions=[]


def source(path):
    path=Path(path)
    if str(path) not in hashes:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    source(path)
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def js(path):source(path);return json.loads(path.read_text(encoding='utf-8-sig'))
def trace(item,panel,path,record,field,value,qualification='Saved content; no new experiment'):
    evidence.append(dict(item=item,panel_or_row=panel,source_path=str(source(path)),source_record=record,source_field=field,source_value=str(value),source_sha256=hashes[str(path)],qualification=qualification))


def export(fig,num):
    fig.savefig(OUT/'figures'/f'figure{num}.pdf',metadata={'Title':f'Microscopy evidence figure {num}','Author':''})
    fig.savefig(OUT/'figures'/f'figure{num}.png',dpi=400)
    plt.close(fig)


def pretty(value):
    s=str(value)
    for a,b in [('unchanged0','unchanged 0'),('a0-1scale','a 0-1 scale'),('uses6446','uses 6446'),('are64x64','are 64 x 64'),('Literal100','Literal 100'),('use18','use 18'),('and20','and 20'),('conditional95','conditional 95'),('among43','among 43'),('control95','control 95'),('guaranteed5','guaranteed 5'),('The0%','The 0%'),('and95','and 95'),('for46','for 46'),('cohort:600','cohort: 600'),('averages10','averages 10'),('source-omission','source-omission'),('JUMP25','JUMP 25'),('atceil80','at ceil 80')]:s=s.replace(a,b)
    s=re.sub(r'(?<=\d)(?=(?:pixels|wells|mixed|test|crops|groups|distinct|nested|pairs|seeds|studies|conditions|TIFFs)\b)',' ',s)
    s=re.sub(r'%(?=[A-Za-z])','% ',s).replace('30APgroups','30 AP groups').replace('bars:16','bars: 16').replace('Bars:100','Bars: 100').replace('Saved95','Saved 95').replace('ceil80','ceil 80').replace('MAE(nM)','MAE (nM)')
    s=re.sub(r';(?=\S)','; ',s)
    return s


def bar(ax,size,label):
    ax.plot([4,4+size],[58,58],color='white',lw=2)
    ax.text(4,54,label,color='white',fontsize=7,va='bottom')


def main():
    rows=read(MM/'SPLIT_MANIFEST.csv');chosen=read(MM/'FIGURE_WELLS.csv')
    fig,axes=plt.subplots(2,3,figsize=(6.875,4.25),layout='constrained')
    for i,w in enumerate(chosen):
        row=next(r for r in rows if r['well']==w['well'] and r['dataset']=='TNBC');path=ROOT/row['provenance'];source(path)
        with h5py.File(path,'r') as f:
            plate=next(iter(f['sample/0/plate']));well=w['well'].split(':')[1];pos=next(iter(f['sample/0/plate/'+plate+'/experiment/'+well+'/position'].values()))
            raw=pos['image/channel'][:,0,0];assert raw.shape==(3,1040,1392)
            names=[v[0].decode() for v in f['definition/feature/tertiary__expanded/object_features'][()]]
            marker=pos['feature/tertiary__expanded/object_features'][()][:,names.index('spotfeatures_count')]
            centers=pos['feature/primary__primary4/center'][()];high=marker>float(row['cutoff'])
            assert len(marker)==int(row['cells']) and int(high.sum())==int(row['positives'])
        for j,label in enumerate(['DAPI input','Cy5 tubulin input',r'Cy3 $\gamma$H2AX readout']):
            ax=axes[i,j];ax.imshow(raw[j],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(('abc'[j] if i==0 else 'def'[j])+'. '+label)
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel(w['well'],fontsize=8)
            if j==2:ax.scatter(centers['x'][high],centers['y'][high],facecolors='none',edgecolors='#ffcb32',s=18,lw=.65)
            ax.plot([70,170],[960,960],color='white',lw=2);ax.text(70,920,'100 px',color='white',fontsize=7)
            trace('Figure 1',f'{i},{j}',path,'position1/image/channel',f'channel {j}',label,'Fixed8bit range0-255; no per-field contrast stretch')
        trace('Figure 1',w['well'],MM/'SPLIT_MANIFEST.csv',str(row),'cutoff;cells;positives',f"{row['cutoff']};{row['cells']};{row['positives']}")
    export(fig,1)
    captions.append(('Figure 1','TNBC microscopy inputs and measured marker readouts. DAPI and Cy5 tubulin are permitted inputs; Cy3 gammaH2AX is withheld from prediction. Yellow rings identify source-defined marker-high cells (spot count>19 for MDA231 and>21 for MDA468) in the preselected held-out C01 wells. All planes use the unchanged0-255 display range. Bars:100pixels. Physical pixel calibration is unverified. These are measured marker states, not independent functional or causal drug-response labels.'))
    reconpath=EDIT/'NF_RECONSTRUCTIONS.npz';source(reconpath);data=np.load(reconpath);meta=js(EDIT/'NF_cells.json');examples=js(EDIT/'EXAMPLE_MANIFEST.json');imagespath=EDIT/'NF_images.npz';source(imagespath);saved=np.load(imagespath)
    fig,axes=plt.subplots(2,5,figsize=(6.875,3.3),layout='constrained')
    for i,e in enumerate(v for v in examples if v['assay']=='NF'):
        idx=e['crop_index'];local=int(np.where(data['indices']==idx)[0][0]);truth=data['original'][local,1].astype(float)
        assert meta[idx]['id']==e['cell']['id'] and np.array_equal(data['original'][local],saved['images'][idx])
        pred=data['unet'][local].astype(float);panels=[data['original'][local,0],data['original'][local,2],truth,pred,np.abs(pred-truth)]
        for j,(name,a) in enumerate(zip(['DAPI input','DHE input',r'NF-$\kappa$B truth','U-Net estimate','Absolute error'],panels)):
            ax=axes[i,j];im=ax.imshow(a,cmap='magma',vmin=0,vmax=1,interpolation='nearest');ax.set_title(name);ax.set_xticks([]);ax.set_yticks([]);bar(ax,16,'16 px')
            if j==0:ax.set_ylabel(meta[idx]['context']+'\n'+meta[idx]['well']+' '+meta[idx]['role'],fontsize=8)
            trace('Figure 2',f'{i},{j}',reconpath,f'indices[{local}]={idx}',['original DAPI','original DHE','original marker','unet','abs(unet-original marker)'][j],meta[idx]['id'],'Display0-1; saved first example per split, not selected by error')
    fig.colorbar(im,ax=axes[:,2:].ravel().tolist(),fraction=.022,pad=.02,label='Normalized intensity / absolute error')
    export(fig,2)
    captions.append(('Figure 2','Withheld NF-kappaB fluorescence reconstruction for the original first held-out-well and held-line examples. DAPI and DHE are observed; NF-kappaB is withheld. Prediction arrays are saved U-Net outputs; error is the absolute pixel difference, without rerunning inference. Marker truth, prediction and error share a0-1scale; source marker normalization uses6446.466000000015 with clipping. Inputs retain their original field normalization. Crops are64x64pixels; bars:16pixels. Physical pixel calibration and subpixel chromatic registration remain unverified. Reconstruction is not counterfactual biological image editing.'))
    mixtures=read(JUMP/'mixture_summary.csv');flags=read(JUMP/'control_flags.csv');methods=['mean','mmd','sliced_w2','local_abundance','region_primary','region_max8'];names=['Mean','MMD','Sliced W2','Local abundance','Local enrichment','Max8 regions']
    fig,axes=plt.subplots(1,2,figsize=(6.875,3.35),layout='constrained');curves=[]
    for k,(method,name) in enumerate(zip(methods,names)):
        own=[r for r in mixtures if r['budget']=='100' and r['construction']=='integer' and float(r['quantile'])==.95 and r['method']==method and float(r['nominal_fraction'])<=.25];own.sort(key=lambda r:float(r['actual_fraction']))
        x=np.array([float(r['actual_fraction'])*100 for r in own]);y=np.array([float(r['mean_flag_rate']) for r in own]);lo=np.array([float(r['ci_low']) for r in own]);hi=np.array([float(r['ci_high']) for r in own]);axes[0].plot(x,y,'o-',ms=3,label=name);axes[0].fill_between(x,lo,hi,alpha=.08)
        f=next(r for r in flags if r['budget']=='100' and r['method']==method and float(r['quantile'])==.95);axes[1].bar(k,float(f['flags'])/float(f['control_wells']));axes[1].text(k,float(f['flags'])/float(f['control_wells'])+.007,f"{f['flags']}/{f['control_wells']}",ha='center',fontsize=7)
        for r in own:
            trace('Figure 3',method,JUMP/'mixture_summary.csv',str({v:r[v] for v in ['budget','method','quantile','construction','nominal_fraction']}),'mean_flag_rate;ci_low;ci_high',f"{r['mean_flag_rate']};{r['ci_low']};{r['ci_high']}")
        trace('Figure 3',method,JUMP/'control_flags.csv',str(f),'flags/control_wells',f"{f['flags']}/{f['control_wells']}")
    axes[0].set(xlabel='Constructed treated fraction (%)',ylabel='Empirical detection fraction',ylim=(-.02,1.02),title='a. Literal 100-cell mixtures');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,fontsize=7);axes[1].set(xticks=range(6),xticklabels=names,ylabel='Held-out control flag fraction',ylim=(0,.3),title='b. Observed control flags');axes[1].tick_params(axis='x',rotation=55)
    export(fig,3)
    captions.append(('Figure 3','JUMP constructed minority-mixture detection and control behavior. Literal100-cell mixtures use18distinct donor/background pairs and20nested cell samples. Shading shows saved conditional95%pair-bootstrap intervals. Control bars show observed flags among43held-out DMSO wells. Thresholds are reference/control95th-percentile rules, not guaranteed5%tests or matched observed error rates. The0%mixtures and DMSO controls are different evaluations. Regions used treatment-informed discovery; spatial treatment-column confounding persists. Constructed mixtures are not naturally measured minority responses.'))
    summary=read(MM/'SUMMARY.csv');robust=read(RGME/'SUMMARY.csv');marker_methods=['random_forest','marker_regression','masked_MLP','MMSL','count_tail'];fig,axes=plt.subplots(1,2,figsize=(6.875,3.35),layout='constrained')
    for method in marker_methods:
        own=[next(r for r in summary if r['dataset']=='TNBC' and r['context']=='pooled' and r['method']==method and r['mask']==mask and r['metric']=='AP') for mask in ['11','10','01']];y=np.array([float(r['estimate']) for r in own]);axes[0].plot(range(3),y,'o-',ms=3,label=method.replace('_',' '));
        for j,r in enumerate(own):axes[0].errorbar(j,y[j],yerr=[[y[j]-float(r['low'])],[float(r['high'])-y[j]]],color='gray',lw=.6,capsize=2);trace('Figure 4a',method,MM/'SUMMARY.csv',str({k:r[k] for k in ['dataset','context','method','mask','metric']}),'estimate;low;high',f"{r['estimate']};{r['low']};{r['high']}")
    for method in ['RF_aug','attention_aug','RGME_no_stability','RGME']:
        r=next(v for v in robust if v['role']=='test' and v['method']==method);axes[1].plot(range(3),[float(r[k]) for k in ['clean_ap','corrupted_ap','worst_ap']],'o-',ms=3,label=method.replace('_',' '));trace('Figure 4b',method,RGME/'SUMMARY.csv',str(r),'clean_ap;corrupted_ap;worst_ap',str([r[k] for k in ['clean_ap','corrupted_ap','worst_ap']]),'Three fit seeds; no pointwise CI supplied by this summary')
    for ax in axes:ax.set(ylim=(0,.7),ylabel='Average precision');ax.legend(fontsize=6.8)
    axes[0].set(xticks=range(3),xticklabels=['Both','DAPI','Tubulin'],title='a. Missing channels\nSpot-count target');axes[1].set(xticks=range(3),xticklabels=['Clean','Mean corrupted','Worst condition'],title='b. Image corruption\nIntensity target');axes[1].tick_params(axis='x',rotation=15)
    export(fig,4)
    captions.append(('Figure 4','TNBC marker prediction under missing channels and input corruption. (a)Saved spot-count-target AP and95%group-bootstrap intervals for46test wells (41mixed-state wells;30APgroups). (b)Saved three-seed mean AP for the distinct intensity-target crop cohort:600test crops/25wells,19mixed-state evaluation wells. Mean corrupted AP averages10fixed conditions; worst AP is the worst condition after averaging seeds. No unavailable per-point intervals are added. These panels use different labels,features and denominators and must not be ranked across panels. Conventional RF/MLP baselines and the unfavorable RGME ablation remain visible.'))

if __name__=='__main__':main()
