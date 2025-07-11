from typing import Iterable, Literal

import numpy as np
from PIL import Image


def label_to_image(label, palette: list[tuple[int, int, int]]) -> np.ndarray:
    label = label.astype(np.uint8)
    arr = np.zeros((*label.shape, 3), dtype=np.uint8)
    for i in range(label.max() + 1):
        arr[label == i] = palette[i]
    return arr


def make_slices(
    arr: np.ndarray,
    axis: Literal["axial", "coronal", "sagittal"],
) -> Iterable[Image.Image]:
    match axis:
        case "axial":
            for i in range(arr.shape[0]):
                yield Image.fromarray(arr[i, :, :])
        case "coronal":
            for i in range(arr.shape[1]):
                yield Image.fromarray(arr[:, i, :])
        case "sagittal":
            for i in range(arr.shape[2]):
                yield Image.fromarray(arr[:, :, i])
        case _:
            raise ValueError("Invalid axis")


def image_slices(
    image: np.ndarray,
    axis: Literal["axial", "coronal", "sagittal"],
) -> Iterable[Image.Image]:
    im = (image * 255).astype(np.uint8)
    yield from make_slices(im, axis)


def label_slices(
    label: np.ndarray,
    axis: Literal["axial", "coronal", "sagittal"],
    palette: list[tuple[int, int, int]],
) -> Iterable[Image.Image]:
    lb = label_to_image(label, palette)
    yield from make_slices(lb, axis)


def image_label_slices(
    image: np.ndarray,
    label: np.ndarray,
    axis: Literal["axial", "coronal", "sagittal"],
    palette: list[tuple[int, int, int]],
    alpha: int = 128,
) -> Iterable[Image.Image]:
    im = (image * 255).astype(np.uint8)
    lb = label_to_image(label, palette)
    for i, j in zip(make_slices(im, axis), make_slices(lb, axis)):
        i = i.convert("RGBA")
        j = j.convert("RGBA")
        j.putalpha(alpha)
        # make background transparent
        arr = np.array(j)
        arr[:, :, 3] = np.where(arr[:, :, :3].sum(axis=2) > 0, alpha, 0)
        j = Image.fromarray(arr)
        yield Image.alpha_composite(i, j).convert("RGB")
