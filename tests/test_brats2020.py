import nibabel as nib
import numpy as np

from src.data.brats import PatientGroupedSampler
from src.data.nifti import NiftiPatient
from src.data.splits import discover_patients


def test_patient_grouped_sampler_keeps_slices_together():
    class SampleDataset:
        samples = [(patient, z) for patient in range(4) for z in range(5)]

        def __len__(self):
            return len(self.samples)

    dataset = SampleDataset()
    sampler = PatientGroupedSampler(dataset, seed=42)
    indices = list(sampler)
    patient_order = [dataset.samples[index][0] for index in indices]
    patient_transitions = sum(
        current != previous
        for previous, current in zip(patient_order, patient_order[1:])
    )

    assert sorted(indices) == list(range(len(dataset)))
    assert patient_transitions == len(set(patient_order)) - 1
    assert list(sampler) == indices


def test_nifti_patient_loader_supports_brats2020_names(tmp_path):
    patient_dir = tmp_path / "BraTS20_Training_001"
    patient_dir.mkdir()

    shape = (16, 16, 8)
    for modality, suffix in {
        "t1": "t1",
        "t1ce": "t1ce",
        "t2": "t2",
        "flair": "flair",
    }.items():
        data = np.random.rand(*shape).astype("float32")
        image = nib.Nifti1Image(data, np.eye(4))
        nib.save(image, patient_dir / f"BraTS20_Training_001_{suffix}.nii.gz")

    seg = np.zeros(shape, dtype="float32")
    seg[4:8, 4:8, 3:5] = 1
    nib.save(nib.Nifti1Image(seg, np.eye(4)), patient_dir / "BraTS20_Training_001_seg.nii.gz")

    patient = NiftiPatient(patient_dir)
    images = patient.load_modalities()
    assert set(images.keys()) == {"t1", "t1ce", "t2", "flair"}
    assert images["t1"].shape == shape
    assert patient.load_segmentation().shape == shape


def test_discover_patients_handles_brats_training_layout(tmp_path):
    dataset_root = tmp_path / "BraTS2020"
    training_root = dataset_root / "Training"
    training_root.mkdir(parents=True)

    patient_dir = training_root / "BraTS20_Training_001"
    patient_dir.mkdir()
    (patient_dir / "BraTS20_Training_001_flair.nii.gz").touch()
    (patient_dir / "BraTS20_Training_001_seg.nii.gz").touch()

    patients = discover_patients(dataset_root)
    assert any(p.name == "BraTS20_Training_001" for p in patients)
