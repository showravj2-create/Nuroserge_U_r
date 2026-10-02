# NeuroSeg-U: Brain Tumor Segmentation with a 2.5D Attention U-Net

A reproducible baseline for **brain tumor segmentation from multi-modal MRI** (BraTS 2020),
built around evaluation that is hard to fool: patient-level splits enforced in code, a
held-out test set, full-volume 3D metrics, and bootstrap confidence intervals.

> **Status: research prototype.** The training and evaluation pipeline is implemented and
> tested. Full experiment results are still to be produced; see [Results](#results).
> Not a diagnostic device.

## Research question

How reliable is a U-Net-based segmentation system when MRI sequences are degraded or
missing, and which design choices (boundary-aware loss, modality dropout, attention gates)
actually help under a controlled, leakage-free protocol?

### Hypotheses (to be tested)

1. A gradient-based boundary term on top of Dice + BCE improves contour quality (HD95).
2. Modality dropout during training improves robustness when a sequence is missing at test time.
3. Attention gates give a measurable gain over a plain U-Net at equal capacity.

## What is implemented

| Component | Status |
|---|---|
| Patient-level train/val/test split, one seed shared by all splits, overlap asserted in code | Implemented |
| Input normalisation from image intensities only (no label information) | Implemented |
| 2.5D input (3 neighbouring slices x 4 modalities), 2D U-Net, optional attention gates | Implemented |
| Dice + BCE + gradient-boundary loss, per-component logging | Implemented |
| Whole-tumour target, or WT / TC / ET regions (`data.target_mode`) | Implemented |
| Modality dropout (training) and modality removal (test-time robustness) | Implemented |
| Patient-level 3D Dice / IoU / sensitivity / precision / HD95, bootstrap 95% CIs | Implemented |
| Run metadata: config, git commit, environment, exact patient split saved per run | Implemented |
| Unit tests covering the failure modes above | Implemented |
| Monte-Carlo dropout / TTA uncertainty maps, calibration analysis | **Planned, not implemented** |
| nnU-Net comparison | **Planned, not implemented** |
| 3D U-Net | **Planned, not implemented** |

## Data

Uses **BraTS 2020**. Obtain it through the official challenge route and follow its licence.
Do **not** commit patient data, NIfTI volumes, predictions, or checkpoints to GitHub
(`.gitignore` excludes them).

Raw labels are `{0, 1, 2, 4}`. `src/data/labels.py` converts them to binary targets:

| Region | Labels |
|---|---|
| WT whole tumour | 1, 2, 4 |
| TC tumour core | 1, 4 |
| ET enhancing tumour | 4 |

## Pipeline

```
Raw NIfTI (t1, t1ce, t2, flair, seg)
  -> keep patients that have a segmentation
  -> patient-level split (one seed, disjointness asserted)
  -> per-modality z-score over non-zero (brain) voxels     [never uses the label]
  -> 2.5D slices resized to 128x128
  -> train: all tumour slices + a fraction of tumour-free slices
  -> U-Net, loss = Dice + BCE (+ boundary)
  -> validation: every slice of every val patient, 3D Dice per patient
  -> test: probabilities resized back to native resolution, 3D metrics, bootstrap CIs
```

## Quick start

```bash
git clone https://github.com/showravj2-create/Nuroserg_U_r
cd Nuroserg_U_r
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

Smoke test with synthetic data (no dataset needed):

```bash
python scripts/train.py --config configs/smoke.yaml
```

BraTS 2020:

```bash
python scripts/check_brats_dataset.py --data-root /path/to/BraTS2020
python scripts/train.py    --config configs/brats_2p5d.yaml --data-root /path/to/BraTS2020
python scripts/evaluate.py --checkpoint runs/brats_2p5d_unet/best.pt --split test
python scripts/evaluate.py --checkpoint runs/brats_2p5d_unet/best.pt --split test --drop-modality flair
```

For Google Colab see [COLAB.md](COLAB.md).

## Experiments

All runs share the same split (`data.split_seed`) and schedule; only the listed factor changes.

| ID | Experiment | How to run | Status |
|---|---|---|---|
| E0 | Plain U-Net, Dice + BCE | `configs/brats_baseline.yaml` | not yet run |
| E1 | + gradient-boundary loss | baseline with `loss.boundary_weight: 0.2` | not yet run |
| E2 | + modality dropout | `data.modality_dropout: 0.10` | not yet run |
| E3 | + attention gates (full model) | `configs/brats_2p5d.yaml` | not yet run |
| E4 | WT / TC / ET regions | `data.target_mode: regions`, `model.out_channels: 3` | not yet run |
| E5 | Missing-sequence robustness | `evaluate.py --drop-modality <name>` on E0 vs E2/E3 | not yet run |

## Evaluation protocol

- **Unit of analysis is the patient.** Metrics are computed on the full 3D volume per
  patient and per region, then averaged across patients. Slice-averaged Dice is not reported:
  it is dominated by empty slices and is not comparable to published BraTS numbers.
- **Empty ground truth.** If a region is absent in a patient (e.g. no enhancing tumour),
  its Dice/HD95 are excluded from that region's mean. The number of such patients and any
  false positives on them are reported alongside.
- **Native resolution.** The network works at 128x128; probability maps are upsampled to the
  native grid before thresholding, and HD95 uses the real voxel spacing (mm).
- **Model selection** uses the validation split only. The test split is touched by
  `evaluate.py --split test`, after model selection.
- **Uncertainty:** 95% bootstrap confidence intervals over patients.

## Results

Not yet available. The table below is filled in from `runs/<experiment>/eval_test.json` after the
experiments above have been run.

| Experiment | WT Dice [95% CI] | TC Dice | ET Dice | WT HD95 (mm) |
|---|---|---|---|---|
| E0 | - | - | - | - |
| E3 | - | - | - | - |

## Limitations

- 2D/2.5D at 128x128 is a deliberate, compute-friendly baseline; it will trail 3D methods
  and nnU-Net. A comparison is planned.
- Single dataset and a single random split; no cross-validation or external validation yet.
- "Boundary loss" here is a gradient-magnitude surrogate, not the signed-distance boundary
  loss of Kervadec et al.
- Retrospective benchmark results do not establish clinical utility.

## Reproducibility

Every training run writes `runs/<experiment>/run.json` with the full config, git commit
(and whether the tree was dirty), library versions, GPU, and the exact list of patients in
each split. `evaluate.py` refuses a `--config` whose split settings differ from the
checkpoint's.

## Ethical and medical disclaimer

For research and education only. Segmentations can be wrong and must not be used for
diagnosis or treatment decisions. Public datasets differ from real clinical populations in
acquisition protocol, demographics, and annotation practice.

## References

- Ronneberger, Fischer & Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015).
- Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas* (2018).
- Isensee et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* (Nature Methods, 2021).
- Menze et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)* (IEEE TMI, 2015); BraTS challenge resources.

## Author

**Showrav Das**: BSc (Hons) Mathematics
