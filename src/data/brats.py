from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, Sampler

from src.data.labels import label_to_targets, num_targets
from src.data.nifti import NiftiPatient
from src.data.preprocessing import normalize_modalities
from src.data.splits import (
    assert_disjoint,
    discover_patients,
    split_manifest,
    split_patients,
)


class PatientGroupedSampler(Sampler):
    """Shuffle patients and slices while keeping each patient's slices together."""

    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        grouped_indices = {}
        for sample_index, (patient_index, _) in enumerate(self.dataset.samples):
            grouped_indices.setdefault(patient_index, []).append(sample_index)

        rng = random.Random(self.seed + self.epoch)
        patient_indices = list(grouped_indices)
        rng.shuffle(patient_indices)
        indices = []
        for patient_index in patient_indices:
            patient_samples = grouped_indices[patient_index]
            rng.shuffle(patient_samples)
            indices.extend(patient_samples)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def assert_no_patient_overlap(*datasets):
    """Raise if any patient appears in more than one of the given datasets."""
    assert_disjoint({f"dataset_{i}({d.split})": d.patient_ids for i, d in enumerate(datasets)})


class Brats2020Dataset(Dataset):
    """2.5D BraTS dataset: axial slices with neighbouring-slice context per modality.

    Inputs are normalised using only the image intensities (never the label).
    Targets are binary {0, 1}: whole tumour, or the three BraTS regions.

    Channel layout is modality-major: channels [i*k, (i+1)*k) belong to modality i,
    where k is the number of context slices.
    """

    def __init__(
        self,
        data_root,
        split="train",
        modalities=None,
        context=3,
        target_size=(240, 240),
        split_seed=42,
        split_fractions=(0.70, 0.15, 0.15),
        max_cached_patients=2,
        target_mode="whole_tumor",
        slice_policy=None,
        negative_fraction=0.25,
        modality_dropout=0.0,
        aug_seed=0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.modality_names = modalities or ["t1", "t1ce", "t2", "flair"]
        self.context = int(context)
        self.slices_per_modality = 1 if self.context <= 1 else self.context
        self.target_size = tuple(target_size)
        self.max_cached_patients = max(1, int(max_cached_patients))
        self.target_mode = target_mode
        self.num_targets = num_targets(target_mode)
        self.modality_dropout = float(modality_dropout)

        if slice_policy is None:
            slice_policy = "tumor_plus_negatives" if split == "train" else "all"
        if slice_policy not in ("all", "tumor_plus_negatives"):
            raise ValueError(f"Unknown slice_policy {slice_policy!r}")
        self.slice_policy = slice_policy

        # Only folders that have ground truth take part in the split.
        patients = discover_patients(self.data_root, require_segmentation=True)
        if len(patients) < 3:
            raise ValueError(
                "BraTS training requires at least 3 patient folders with MRI volumes "
                f"and a segmentation. Found {len(patients)} under {self.data_root}."
            )

        train_fraction, val_fraction, test_fraction = split_fractions
        splits = split_patients(
            patients,
            train_ratio=train_fraction,
            val_ratio=val_fraction,
            test_ratio=test_fraction,
            seed=split_seed,
        )
        assert_disjoint(splits)

        if split not in splits:
            raise ValueError(f"split must be one of {sorted(splits)}, got {split!r}")

        self.split_manifest = split_manifest(splits)
        self.patients = splits[split]
        self.patient_ids = [p.name for p in self.patients]

        self._patient_cache = OrderedDict()
        self._np_rng = np.random.default_rng(aug_seed)
        slice_rng = random.Random(aug_seed)

        self.depths = []
        self.samples = []
        for patient_index, patient_dir in enumerate(self.patients):
            label = NiftiPatient(patient_dir).load_segmentation()
            depth = label.shape[2]
            self.depths.append(depth)

            has_tumor = (np.rint(label) > 0).any(axis=(0, 1))
            for z in range(depth):
                keep = (
                    self.slice_policy == "all"
                    or bool(has_tumor[z])
                    or slice_rng.random() < negative_fraction
                )
                if keep:
                    self.samples.append((patient_index, z))

        if not self.samples:
            raise ValueError(
                f"No usable slices were found in {self.data_root} for the {split} split. "
                "Check the BraTS folder layout or modality names."
            )

    @classmethod
    def from_config(cls, cfg, split, data_root=None):
        """Build a dataset from a config dict.

        Train, validation and evaluation all go through this method so that they
        are guaranteed to use the same split seed and fractions.
        """
        d = cfg["data"]
        return cls(
            data_root=data_root or d["root"],
            split=split,
            modalities=d.get("modalities"),
            context=d.get("num_slices_context", 3),
            target_size=tuple(d.get("target_size", [240, 240])),
            split_seed=d.get("split_seed", cfg["seed"]),
            split_fractions=(
                d.get("train_fraction", 0.70),
                d.get("val_fraction", 0.15),
                d.get("test_fraction", 0.15),
            ),
            max_cached_patients=d.get("max_cached_patients", 2),
            target_mode=d.get("target_mode", "whole_tumor"),
            slice_policy=d.get("slice_policy"),
            negative_fraction=d.get("negative_slice_fraction", 0.25),
            modality_dropout=d.get("modality_dropout", 0.0),
            aug_seed=cfg["seed"] + 1,
        )

    @property
    def in_channels(self):
        return len(self.modality_names) * self.slices_per_modality

    def channel_slice(self, modality):
        """Channel range for a modality given by name or index."""
        if isinstance(modality, str):
            index = self.modality_names.index(modality)
        else:
            index = int(modality)
        k = self.slices_per_modality
        return slice(index * k, (index + 1) * k)

    # ------------------------------------------------------------------
    # slice construction
    # ------------------------------------------------------------------

    def _slice_indices(self, depth, center):
        if self.context <= 1:
            return [min(max(center, 0), depth - 1)]

        indices = []
        for offset in range(self.context):
            idx = int(round(center - (self.context - 1) / 2 + offset))
            idx = min(max(idx, 0), depth - 1)
            indices.append(idx)
        return indices

    def _resize_volume(self, array, is_label=False):
        if array.ndim == 2:
            array = array[None, :, :]

        if array.shape[-2:] == self.target_size:
            return array.astype(np.float32)

        target_h, target_w = self.target_size
        zoom_y = target_h / array.shape[-2]
        zoom_x = target_w / array.shape[-1]
        out = ndimage.zoom(array, (1.0, zoom_y, zoom_x), order=0 if is_label else 1)
        if out.shape[-2:] != self.target_size:
            out = np.pad(
                out,
                (
                    (0, 0),
                    (0, max(0, target_h - out.shape[-2])),
                    (0, max(0, target_w - out.shape[-1])),
                ),
                mode="edge",
            )
            out = out[:, :target_h, :target_w]
        return out.astype(np.float32)

    def _build_input(self, images, z):
        stacks = []
        for modality in self.modality_names:
            volume = images[modality]
            indices = self._slice_indices(volume.shape[2], z)
            stacks.append(np.stack([volume[:, :, i] for i in indices], axis=0))

        x = np.concatenate(stacks, axis=0).astype(np.float32)
        return self._resize_volume(x)

    def _build_target(self, label, z):
        targets = label_to_targets(label[:, :, z], self.target_mode)  # (K, H, W) in {0,1}
        return self._resize_volume(targets, is_label=True)

    def _apply_modality_dropout(self, x):
        n = len(self.modality_names)
        drop = self._np_rng.random(n) < self.modality_dropout
        if drop.all():
            drop[int(self._np_rng.integers(n))] = False  # never drop every modality
        for index, dropped in enumerate(drop):
            if dropped:
                x[self.channel_slice(index)] = 0.0  # 0 is the z-score mean
        return x

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        patient_index, z = self.samples[index]
        images, label = self._load_patient(patient_index)

        x = self._build_input(images, z)
        y = self._build_target(label, z)

        if self.modality_dropout > 0 and self.split == "train":
            x = self._apply_modality_dropout(x)

        return torch.from_numpy(x), torch.from_numpy(y)

    def _load_patient(self, patient_index):
        if patient_index in self._patient_cache:
            images, label = self._patient_cache.pop(patient_index)
            self._patient_cache[patient_index] = (images, label)
            return images, label

        patient = NiftiPatient(self.patients[patient_index])
        images = patient.load_modalities(self.modality_names)
        label = np.rint(patient.load_segmentation()).astype(np.int16)

        for name, volume in images.items():
            if volume.shape != label.shape:
                raise ValueError(
                    f"Modality {name} shape {volume.shape} does not match label "
                    f"{label.shape} for patient {self.patient_ids[patient_index]}."
                )

        # Normalise from image intensities only. Never derive this mask from `label`.
        images = normalize_modalities(images)

        self._patient_cache[patient_index] = (images, label)
        while len(self._patient_cache) > self.max_cached_patients:
            self._patient_cache.popitem(last=False)
        return images, label

    def load_patient_volume(self, patient_index):
        """Every slice of one patient, for patient-level (3D) evaluation.

        Returns x (D, C, h, w) and y (D, K, h, w) at the training resolution, plus
        the native-resolution integer label (H, W, D) and voxel spacing in mm.
        """
        images, label = self._load_patient(patient_index)
        depth = label.shape[2]

        x = np.stack([self._build_input(images, z) for z in range(depth)], axis=0)
        y = np.stack([self._build_target(label, z) for z in range(depth)], axis=0)

        spacing = NiftiPatient(self.patients[patient_index]).voxel_spacing(
            self.modality_names[0]
        )

        return {
            "patient": self.patient_ids[patient_index],
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "label": label,
            "spacing": spacing,
        }
