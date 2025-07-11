from pathlib import Path

import h5py
import numpy as np

from ..domain import DomainDataset, Stage


class AbdomenCT(DomainDataset):

    names = ["amos_ct1", "amos_ct2", "amos_ct3", "btcv", "tcia", "word"]

    def __init__(
        self,
        data_dir: str | Path,
        domain: int,
        stage: Stage,
        *args,
        **kwargs,
    ):
        # we do not consider train/val stage
        # since dg evaluates on the target domain
        # whose samples are not seen during training
        domain_dir = Path(data_dir) / f"{self.names[domain]}"
        if stage == "train-single":
            stage_dir = domain_dir / "train"
        elif stage == "val-single":
            stage_dir = domain_dir / "val"
        else:
            stage_dir = domain_dir
        if not stage_dir.exists():
            stage_dir = domain_dir
        DomainDataset.__init__(
            self,
            data_dir=stage_dir,
            domain=domain,
            stage=stage,
            image_filter=AbdomenCT.extract_h5_id,
            label_filter=AbdomenCT.extract_h5_id,
        )

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
