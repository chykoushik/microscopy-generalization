# Microscopy generalization

Research code, computational results, and Python-generated figures for **Microscopy generalization**.

The experiments compare established population statistics, marker classifiers, paired fluorescence reconstruction, robustness baselines, and selective ATP evaluation. Wells are the experimental units; cells, crops, sampling repeats, seeds, and corruption views remain nested observations. Acquisition plates and historical study labels do not establish independent biological preparations.

## Datasets

- JUMP Cell Painting: 38,251 encoded cells from 238 wells on three acquisition plates. The minority experiment uses 18 distinct donor/background pairs and literal 100-cell mixtures. Input URLs and well roles are in `data/manifests/jump/`.
- TNBC gammaH2AX measured-marker prediction: 8,450 cells from 131 wells, two lines/plates. DAPI and Cy5 tubulin features exclude the Cy3 marker. Operational source thresholds are 19 spots for MDA231 and 21 for MDA468. The test contains 46 wells, including 41 mixed-state wells; AP uncertainty uses the saved 30 groups.
- NF-kappaB paired reconstruction: 1,971 crops from 92 wells across four acquisition plates; DAPI/DHE inputs predict the withheld marker. SUM159PT is the held-line context.
- GammaH2AX paired reconstruction and corruption: 1,646 crops from 71 wells. This uses nuclear marker intensity, a different endpoint from the spot-count experiment. MDA468 is the held-line context.
- BBBC013: DNA-only prediction of a withheld FKHR-GFP readout, supporting missing-channel evaluation.
- NViR: same-well ATP outcomes, 241 treated wells from six repeatedly inspected technical studies. ATP is signed calibrated nM; TNBC identity and independent biological preparation status are unverified.

Raw microscopy images, original crop arrays, third-party encoder weights, and fitted checkpoints are not distributed. Source acquisition ledgers and well manifests describe expected inputs. Obtain those assets under their source terms before image extraction, retraining, or image-panel regeneration. See `ATTRIBUTION.txt`.

## Installation

Use Python 3.12 from the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
```

PyTorch CPU execution is sufficient for numerical replay. A matching CUDA build can accelerate the small reconstruction models. The JUMP extraction configuration also requires the original DeepCellMorpho architecture source and checkpoint at its configured paths, acquired separately under their licences.

## Experiments and reproduction

Run these commands in a fresh clone. Preserve the distributed result tables before recomputing them. Full training is not implied by the numerical replay commands.

JUMP: `scripts/jump_extract.py` supplies source download and frozen encoding; `scripts/jump_baselines.py` fits reference-only preprocessing and global baselines. With the separately acquired images, encoder assets, and generated cell embeddings present:

```powershell
python scripts/jump_extract.py download
python scripts/jump_extract.py extract
python scripts/jump_baselines.py
python scripts/jump_minority.py evaluate
```

The saved `regions.npz` and split preserve the recorded discovery selection. For a new discovery run use a fresh output location in `scripts/jump_minority.py`; `discover` refuses to overwrite existing discovery results. Results remain conditional on fixed regions, references, and thresholds.

Marker prediction and missing-channel robustness use `scripts/missing_marker.py`, `scripts/missing_marker_stress.py`, and `src/evaluation/missing_marker.py`. Obtain TNBC CellH5 inputs according to `outputs/results/tnbc_source/TNBC_DOWNLOAD_LEDGER.csv`. BBBC013 can be acquired by the supplied downloader. Fresh training and evaluation:

```powershell
python scripts/acquire_missing_marker.py --output outputs/results/marker_replay
python scripts/missing_marker.py --stage baseline --output outputs/results/marker_replay
python scripts/missing_marker.py --stage develop --output outputs/results/marker_replay
python scripts/missing_marker.py --stage evaluate --output outputs/results/marker_replay
python scripts/missing_marker_stress.py --output outputs/results/marker_replay
```

The saved-result tables and numeric prediction arrays are included. Fresh training writes its own code hashes and frozen checkpoint selection; historical checkpoint blobs are not required or distributed. Plotting scripts retain the source image/model prerequisites for localization panels.

For paired reconstruction, place source TIFF/CellH5 files at the paths in `outputs/results/tnbc_spatial_inputs/`, with NF segmentations generated using `segment` in `src/evaluation/spatial_response_data.py`. Source image IDs and label joins must match those manifests. Preserve the distributed reconstruction outputs, then run:

```powershell
python scripts/prepare_reconstruction.py
python scripts/train_reconstruction.py
python scripts/evaluate_reconstruction.py
python scripts/reconstruction_summary.py
python scripts/corruption_robustness.py
```

The reconstruction preparation refuses existing extracted pixels; training resumes missing checkpoints. Reconstruction code includes only observed-channel reconstruction, excluding alternative-state editing. Corruption evaluates three seeds, five corruption families at two severities, and an intensity target; it does not reuse the spot-count label as the same endpoint.

ATP selection can be replayed directly from the distributed frozen source-risk matrices, prediction errors, and selection indices:

```powershell
python scripts/evaluate_atp.py
```

This re-solves the original mixed-integer selection and checks all recorded per-study MAEs. It does not retrain the historical ATP image predictor. `src/evaluation/atp_risk.py` contains the retained kernel-risk, grouped fitting, source-omission, and selection computations.

## Results

| Experiment | Recorded result |
|---|---|
| JUMP 25% constructed minority | Primary region sensitivity 0.8417 versus MMD 0.0278; control flags 4/43 versus 1/43 |
| GammaH2AX spot-count prediction | Random forest AP 0.4837 and AUROC 0.904; MMSL AP 0.4317 |
| NF-kappaB reconstruction | Foreground MAE 0.08917, SSIM 0.8163, marker AUROC 0.9443 on held wells; held-line AUROC 0.7921 |
| GammaH2AX reconstruction | U-Net foreground MAE 0.02593; nearest-structural baseline has lower foreground error |
| GammaH2AX corruption | Augmented RF AP 0.431 versus RGME 0.344; paired difference -0.087, saved 95% interval approximately [-0.158, -0.020] |
| ATP selection | Joint macro MAE 136.147 nM versus source mean 139.649 at 194/241 retained wells; worst-study error 205.109 versus 203.369 nM |

Full-precision source measurements are under `outputs/results/`; manuscript table exports are in `results/tables/`. Intervals retain their original conditional/grouped meaning. JUMP flags are not matched observed false-positive rates. Constructed fractions are not measured biological responder fractions. Reconstruction is not a counterfactual treatment effect. ATP gains are not uniform across studies; unfavorable subgroup and omission results are retained. No new-method superiority or independent biological generalization is established.

## Generated figures

`figures/figure1.png` shows measured gammaH2AX inputs/readouts; `figure2.png` shows withheld NF-kappaB reconstruction; `figure3.png` compares minority sensitivity and control flags; `figure4.png` separates missing-channel and corruption results. All four are Python-generated experimental figures.

`python scripts/generate_figures.py` regenerates them under `outputs/results/manuscript_figures/` from the numerical tables and separately acquired source image/reconstruction arrays. Figures 1 and 2 require those pixel assets; they are not silently regenerated from synthetic images. Pixel bars do not imply verified micron calibration.

## Rights

Original source code is published without selecting a new software licence; copyright remains with its owner. No permission beyond applicable law and GitHub platform terms is implied. Dataset/model licences remain separate. See `ATTRIBUTION.txt` for sources and restrictions.
