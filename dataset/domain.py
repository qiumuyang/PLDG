from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional

import numpy as np
from torch.utils.data import Dataset

Stage = Literal["train", "val", "train-single", "val-single"]
FileFilter = Callable[[Path], Optional[str]]


class DataItem(NamedTuple):
    image: np.ndarray
    label: np.ndarray
    path: str


class DomainDataset(Dataset):
    """An image segmentation dataset for one domain.

    Attributes:
        data_dir: the directory of the dataset
        domain: the domain id (0-based)
        stage: the stage of the dataset, "train" or "val"
        image_filter: the file filter to select images
        label_filter: the file filter to select labels
            A file filter should take a `Path` object and return an identifier
            (str) if the file is a valid image, otherwise None. The identifiers
            of image and label in the same pair should match.
        image_paths: the paths of the images
        label_paths: the paths of the labels
        path_loaded: whether the paths are loaded

    Loading Process:
        1. Paths of the images and labels are not loaded until `__getitem__` or
           `__len__` is called.
        2. Loading paths is done by `load_path` method. Files are selected by
           the file filters.
        3. Once the paths are loaded, they will be cached and will not change
           unless `load_path` is called again.
        4. Subclasses should implement `load_image_label` method to load image
           and label from a given path. `load_image_label` will be called by
           `__getitem__`.
    """

    stage: Stage

    def __init__(
        self,
        data_dir: str | Path,
        domain: int,
        stage: Stage,
        image_filter: FileFilter | None = None,
        label_filter: FileFilter | None = None,
    ):
        self.data_dir = data_dir
        self.domain = domain
        self.stage = stage

        if image_filter is None or label_filter is None:
            raise ValueError("image_filter and label_filter cannot be None")

        self.image_filter = image_filter
        self.label_filter = label_filter
        self.image_paths = []
        self.label_paths = []
        self.path_loaded = False

        # if stage not in ["train", "test", "val"]:
        #     raise ValueError(
        #         f"stage must be 'train', 'test' or 'val', got {stage}")

    def load_path(self, *args, **kwargs) -> None:
        """Load path info according to the filter and the data directory.

        """
        data_dir = Path(self.data_dir)
        self.image_paths = [
            p for p in data_dir.rglob("*")
            if self.image_filter(p) and p.is_file()
        ]
        self.label_paths = [
            p for p in data_dir.rglob("*")
            if self.label_filter(p) and p.is_file()
        ]
        self.image_paths.sort()
        self.label_paths.sort()

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(
                "number of images and labels mismatch: {} != {}".format(
                    len(self.image_paths), len(self.label_paths)))
        # check identifier match
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            image_id = self.image_filter(image_path)
            label_id = self.label_filter(label_path)
            if image_id != label_id:
                raise ValueError("image identifier mismatch: {} != {}".format(
                    image_id, label_id))

        # check successfully loaded
        if len(self.image_paths) == 0:
            raise ValueError("No dataset paths loaded from {}".format(
                self.data_dir))
        self.path_loaded = True

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load image and label from the given paths."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> DataItem:
        """(n_ch, h, w), (n_cl, h, w)"""
        if not self.path_loaded:
            self.load_path()
        image_path, label_path = (self.image_paths[index],
                                  self.label_paths[index])
        image, label = self.load_image_label(image_path, label_path)
        return DataItem(image=image, label=label, path=str(image_path))

    def get_path(self, index: int) -> str:
        if not self.path_loaded:
            self.load_path()
        return str(self.image_paths[index])

    def __len__(self) -> int:
        if not self.path_loaded:
            self.load_path()
        return len(self.image_paths)

    def __repr__(self) -> str:
        parts = "/".join([
            self.__class__.__name__,
            f"Domain{self.domain}",
            self.stage,
        ])
        return (f"{parts}: {Path(self.data_dir).as_posix()} "
                f"({len(self)})")
