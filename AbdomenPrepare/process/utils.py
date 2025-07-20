import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pydicom
import SimpleITK as sitk


def read_dicom_zip_to_numpy(zip_path: str | Path) -> np.ndarray:
    """
    Reads a zip file containing DICOM slices for one patient and returns a NumPy array.

    Args:
        zip_path (str): Path to the zip file containing DICOM slices.

    Returns:
        numpy.ndarray: 3D NumPy array with shape (depth, height, width).
    """
    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(temp_dir)
        reader.SetFileNames(dicom_files)

        image = reader.Execute()
        return sitk.GetArrayFromImage(image)


def dicom_zip_to_nii(zip_path: str | Path) -> Path:
    """
    Converts a DICOM series (zip) to a NIfTI file.

    Args:
        dicom_path (str): Path to the DICOM series (zip).
        nii_path (str): Path to save the NIfTI file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(temp_dir)
        reader.SetFileNames(dicom_files)

        image = reader.Execute()
        # create a temp nii file
        ret, nii_path = tempfile.mkstemp(suffix=".nii.gz", prefix="dcm")
        if ret < 0:
            raise RuntimeError("Failed to create a temporary NIfTI file.")
        sitk.WriteImage(image, nii_path)
        return Path(nii_path)


def read_dicom_zip_metadata(zip_path: str | Path) -> dict:
    """
    Reads a zip file containing DICOM slices for one patient and returns metadata.

    Args:
        zip_path (str): Path to the zip file containing DICOM slices.

    Returns:
        dict: Metadata dictionary.
    """
    keys = [
        "Manufacturer", "PixelSpacing", "SliceThickness", "InstitutionName"
    ]
    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        ds = pydicom.dcmread(next(Path(temp_dir).rglob("*.dcm")))
        return {key: ds.get(key, "N/A") for key in keys}


def find_bounding_box(
    label: np.ndarray,
    margin: int | tuple[int, int] | tuple[int, int, int, int] = 20
) -> tuple[int, int, int, int]:
    """
    Finds the bounding box of the non-background region in the label.

    Args:
        label (numpy.ndarray): 3D label array with shape (d, h, w).

    Returns:
        tuple: (t, b, l, r) where
               t - top index (inclusive),
               b - bottom index (exclusive),
               l - left index (inclusive),
               r - right index (exclusive).
    """
    if isinstance(margin, int):
        margin = (margin, margin, margin, margin)
    elif len(margin) == 2:  # vertical, horizontal
        margin = (margin[0], margin[0], margin[1], margin[1])

    # Project along the depth axis to get a 2D mask
    projection = np.max(label, axis=0)  # Shape: (h, w)

    # Find indices of non-zero rows and columns
    rows = np.any(projection > 0, axis=1)
    cols = np.any(projection > 0, axis=0)

    # Get the bounding box indices
    t = np.where(rows)[0][0]  # Top
    b = np.where(rows)[0][-1] + 1  # Bottom (exclusive)
    l = np.where(cols)[0][0]  # Left
    r = np.where(cols)[0][-1] + 1  # Right (exclusive)

    tm, bm, lm, rm = margin
    t = max(t - tm, 0)
    b = min(b + bm, projection.shape[0])
    l = max(l - lm, 0)
    r = min(r + rm, projection.shape[1])

    return t, b, l, r


def find_axial_bound(label: np.ndarray,
                     margin: int | tuple[int, int] = 5) -> tuple[int, int]:
    """
    Finds the axial bounds of the non-background region in the label.

    Args:
        label (numpy.ndarray): 3D label array with shape (d, h, w).

    Returns:
        tuple: (inferior, superior) bounds.
    """
    if isinstance(margin, int):
        margin = (margin, margin)

    # Find indices of non-zero slices
    non_zero_slices = np.any(label > 0, axis=(1, 2))

    # Get the first and last non-zero slice indices
    inf = np.where(non_zero_slices)[0][0]
    sup = np.where(non_zero_slices)[0][-1] + 1

    inf = max(inf - margin[0], 0)
    sup = min(sup + margin[1], label.shape[0])

    return inf, sup


def dump_as_nii(data: np.ndarray, path: str | Path):
    """
    Dumps a 3D NumPy array as a NIfTI file.

    Args:
        data (numpy.ndarray): 3D NumPy array.
        path (str): Path to save the NIfTI file.

    Note:
        Spacing and other metadata are not saved.
    """
    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, str(path))


def dump_as_h5(image: np.ndarray, label: np.ndarray, path: str | Path):
    """
    Dumps image and label as a single HDF5 file.

    Args:
        image (numpy.ndarray): 3D image array.
        label (numpy.ndarray): 3D label array.
        path (str): Path to save the HDF5 file.
    """

    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=image, compression="lzf")
        f.create_dataset("label", data=label, compression="lzf")
