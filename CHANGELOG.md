# Changelog

## Unreleased: correctness fixes (results from earlier runs are invalid)

Any Dice or loss numbers produced before this change (for example a validation Dice of
~0.97-0.98 with a strongly negative loss) were **artefacts of bugs, not model quality**, and
should not be reported. Retrain from scratch.

### Fixed
- **Label leakage into the input.** Volumes were normalised inside the ground-truth tumour
  mask and every voxel outside it was set to zero, so the network only saw the tumour.
  Normalisation now uses image intensities only. (`src/data/brats.py`)
- **Non-binary targets.** Raw BraTS labels (0/1/2/4) were fed to BCE/Dice, making the loss
  unbounded below (large negative values). Targets are now binary via `src/data/labels.py`,
  and the first training batch is checked.
- **Train/val patient overlap.** Train and validation used different split seeds (42 vs 43),
  i.e. different shuffles, so patients appeared in both. All splits now share
  `data.split_seed`, and overlap is a hard error.
- **Evaluation used the wrong split.** `evaluate.py` used yet another seed and the
  validation split; the test split was never evaluated. It now evaluates val or test with the
  training split definition.
- **Inflated Dice.** Slice-level Dice averaged over mostly-empty slices, with empty-vs-empty
  scored as ~1. Evaluation is now per patient on the full 3D volume, per region.
- **Ignored `--dataset-root`** in `colab_run.py`.
- **Attention gate shape check** compared a tuple to an int (`x.shape[-2]`).
- `modality_dropout` was in the config but never used; it is now implemented.
- Patients without a segmentation are excluded before splitting instead of silently
  shrinking the val/test sets.

### Added
- WT / TC / ET target mode, patient-level metrics with bootstrap CIs, voxel-spacing-aware 3D HD95.
- Per-run `run.json` (config, git commit, environment, patient split), `history.json`,
  checkpoints under `runs/<experiment>/`.
- Tests for each bug above.
- `configs/brats_baseline.yaml` for ablations.

### Changed
- Training uses all tumour slices plus 25% of tumour-free slices (`data.slice_policy`),
  batch size 8, gradient clipping. Loss is computed in float32 under AMP.
- README now separates implemented features from planned ones.

### Removed
- `v2/` (stale duplicate of the root project), `src/metrics/convblock.py` (broken duplicate
  of the model code), empty `src/segmentation/`, and unused dependencies.
