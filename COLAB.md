# Running NeuroSeg-U on Google Colab

Use a GPU runtime (Runtime → Change runtime type → T4 GPU).

## 1) Clone and install

```python
!git clone https://github.com/showravj2-create/Nuroserg_U_r.git
%cd Nuroserg_U_r
!pip install -q -r requirements.txt
```

## 2) Get BraTS 2020

Download BraTS 2020 from the official challenge or a mirror you are licensed to use,
and extract it so that one folder contains the patient folders:

```text
/content/BraTS2020/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii(.gz)
    BraTS20_Training_001_t1.nii(.gz)
    BraTS20_Training_001_t1ce.nii(.gz)
    BraTS20_Training_001_t2.nii(.gz)
    BraTS20_Training_001_seg.nii(.gz)
```

Any nesting works (e.g. `Training/…`): patient folders are found recursively. Folders
without a `*_seg` file (such as the BraTS validation set) are ignored.

## 3) Validate the dataset and run the tests

```python
!python scripts/check_brats_dataset.py --data-root /content/BraTS2020
!pytest -q
```

The check prints the label values it finds (expected: 0, 1, 2, 4), which patients are
usable, and the split sizes.

## 4) Train

```python
!python scripts/train.py --config configs/brats_2p5d.yaml --data-root /content/BraTS2020
```

Outputs go to `runs/<experiment>/`: `best.pt`, `last.pt`, `history.json`, and
`run.json` (config, git commit, environment, and the exact patient split).

**Colab sessions end.** Copy `runs/` to Drive when training finishes (or point
`--output-dir` at a Drive folder):

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r runs /content/drive/MyDrive/neuroseg_runs
```

## 5) Evaluate on the held-out test split

```python
!python scripts/evaluate.py --checkpoint runs/brats_2p5d_unet/best.pt --split test
```

This predicts every slice of each test patient, resizes the probabilities back to the
native image grid, and reports 3D Dice/IoU/HD95 per patient with bootstrap confidence
intervals. Add `--drop-modality flair` to measure robustness to a missing sequence.

## Notes

- If Colab runs out of memory, lower `training.batch_size` in the config.
- To smoke-test the pipeline without data: `python scripts/train.py --config configs/smoke.yaml`.
- Keep `data.split_seed` and the split fractions fixed across every run you want to compare.
