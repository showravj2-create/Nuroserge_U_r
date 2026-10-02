import numpy as np
import nibabel as nib

from src.data.nifti import NiftiPatient


def test_nifti_patient_loader(
    tmp_path
):

    patient_dir = (
        tmp_path / "BraTS-test"
    )

    patient_dir.mkdir()

    shape = (16, 16, 8)

    for modality in [
        "t1n",
        "t1c",
        "t2w",
        "t2f"
    ]:

        data = np.random.rand(
            *shape
        ).astype("float32")

        image = nib.Nifti1Image(
            data,
            np.eye(4)
        )

        nib.save(
            image,
            patient_dir
            / f"BraTS-test_{modality}.nii.gz"
        )

    segmentation = np.zeros(
        shape,
        dtype="float32"
    )

    segmentation[
        4:8,
        4:8,
        3:5
    ] = 1

    image = nib.Nifti1Image(
        segmentation,
        np.eye(4)
    )

    nib.save(
        image,
        patient_dir
        / "BraTS-test_seg.nii.gz"
    )

    patient = NiftiPatient(
        patient_dir
    )

    images = patient.load_modalities()

    mask = patient.load_segmentation()

    assert set(images.keys()) == {
        "t1n",
        "t1c",
        "t2w",
        "t2f"
    }

    assert images["t1n"].shape == shape

    assert mask.shape == shape


def test_nifti_patient_loader_supports_uncompressed_files(tmp_path):
    patient_dir = tmp_path / "BraTS-nii"
    patient_dir.mkdir()
    shape = (8, 8, 4)

    for modality in ["t1n", "t1c", "t2w", "t2f"]:
        data = np.random.rand(*shape).astype("float32")
        nib.save(
            nib.Nifti1Image(data, np.eye(4)),
            patient_dir / f"BraTS-nii_{modality}.nii",
        )

    segmentation = np.zeros(shape, dtype="float32")
    nib.save(
        nib.Nifti1Image(segmentation, np.eye(4)),
        patient_dir / "BraTS-nii_seg.nii",
    )

    patient = NiftiPatient(patient_dir)
    assert set(patient.load_modalities()) == {"t1n", "t1c", "t2w", "t2f"}
    assert patient.load_segmentation().shape == shape