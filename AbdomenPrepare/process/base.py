from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .utils import find_axial_bound, find_bounding_box


class Processor(ABC):

    margin_d: tuple[int, int]
    margin_h: tuple[int, int]
    margin_w: tuple[int, int]
    window: tuple[int, int]
    has_validation: bool

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = [
            "margin_d", "margin_h", "margin_w", "window", "has_validation"
        ]
        for attr in required:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} must have {attr}")

    def __init__(self, data_dir: str | Path, class_mapping: list[int]):
        self.data_dir = Path(data_dir)
        self.class_mapping = np.array(class_mapping)

    def get_bounding_box(
        self,
        path: Path,
        image: np.ndarray,
        label: np.ndarray,
    ) -> tuple[int, int, int, int, int, int]:
        """For cropping ROI from the image and label.

        (d1, d2, h1, h2, w1, w2) = get_bounding_box(path, image, label)
        """
        t, b, l, r = find_bounding_box(label,
                                       (self.margin_h[0], self.margin_h[1],
                                        self.margin_w[0], self.margin_w[1]))
        i, s = find_axial_bound(label, (self.margin_d[0], self.margin_d[1]))
        return i, s, t, b, l, r

    def get_dataset_name(self) -> str:
        """Return the dataset name"""
        return self.__class__.__name__.removesuffix("Processor").lower()

    @abstractmethod
    def load_raw(
        self,
        stage: Literal["train", "val"] = "train",
    ) -> Iterable[tuple[Path, np.ndarray, np.ndarray]]:
        """Load the raw data"""
        ...

    def load(
        self,
        stage: Literal["train", "val"] = "train",
    ) -> Iterable[tuple[Path, np.ndarray, np.ndarray]]:
        """Load the data and normalize it"""
        for path, image, label in self.load_raw(stage):
            print(path.as_posix())
            label = self.class_mapping[label.astype(np.int32)]

            inf, sup, ant, post, left, right = self.get_bounding_box(
                path, image, label)
            image = image[inf:sup, ant:post, left:right]
            label = label[inf:sup, ant:post, left:right]

            yield path, image, label
