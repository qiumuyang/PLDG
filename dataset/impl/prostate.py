from pathlib import Path

import numpy as np

from ..domain import DomainDataset, Stage


class Prostate(DomainDataset):
    """Prostate dataset. See https://liuquande.github.io/SAML/."""

    names = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]

    def __init__(self, root: str | Path, domain: int, stage: Stage):
        sub_dir = "train" if stage == "train" else "test"
        DomainDataset.__init__(
            self,
            data_dir=Path(root) / f"{self.names[domain]}" / sub_dir,
            domain=domain,
            stage=stage,
            image_filter=Prostate._prostate_image_filter,
            label_filter=Prostate._prostate_label_filter,
        )

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        image, label = np.load(str(image_path)), np.load(str(label_path))
        # [-1, 1] to [0, 1]
        image = (image + 1) / 2
        return image, label[np.newaxis, ...]

    @staticmethod
    def _prostate_image_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "image":
            return volume + "_" + slice

    @staticmethod
    def _prostate_label_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "label":
            return volume + "_" + slice
