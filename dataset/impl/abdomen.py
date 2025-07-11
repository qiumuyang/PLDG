from pathlib import Path

import h5py
import numpy as np

from ..domain import DomainDataset, Stage


class Abdomen(DomainDataset):

    names = [
        "amos_ct1", "amos_ct2", "amos_ct3", "amos_mr", "btcv", "tcia", "word"
    ]
    no_official_validation = ("btcv", "tcia")
    num_val = 3

    def __init__(
        self,
        data_dir: str | Path,
        domain: int,
        stage: Stage,
        *args,
        **kwargs,
    ):
        if self.names[domain] in self.no_official_validation:
            sub_dir = "train"  # no validation set
        else:
            sub_dir = "train" if stage == "train" else "val"

        DomainDataset.__init__(
            self,
            data_dir=Path(data_dir) / f"{self.names[domain]}" / sub_dir,
            domain=domain,
            stage=stage,
            image_filter=Abdomen.extract_h5_id,
            label_filter=Abdomen.extract_h5_id,
        )

        if self.names[domain] in self.no_official_validation:
            self.load_path()
            image_tr, image_va = (self.image_paths[:-self.num_val],
                                  self.image_paths[-self.num_val:])
            label_tr, label_va = (self.label_paths[:-self.num_val],
                                  self.label_paths[-self.num_val:])
            self.image_paths = image_tr if stage == "train" else image_va
            self.label_paths = label_tr if stage == "train" else label_va

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """

        For h5, the paths of image and label are the same.
        """
        with h5py.File(image_path, "r") as f:
            image: np.ndarray = f["image"][:]  # type: ignore
            label: np.ndarray = f["label"][:]  # type: ignore
        image = image[np.newaxis, ...]
        return image, label

    @staticmethod
    def extract_h5_id(path: Path) -> str | None:
        if path.name.endswith(".h5"):
            return path.name.removesuffix(".h5")
