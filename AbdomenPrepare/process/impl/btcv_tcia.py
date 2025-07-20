from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

from numpy import ndarray
from typing_extensions import override

import process.transforms as T

from ..base import Processor
from ..meta import btcv_class_mapping
from ..utils import dicom_zip_to_nii


class BTCVProcessor(Processor):

    margin_d = (10, 10)
    margin_h = (10, 10)
    margin_w = (20, 20)
    window = (-125, 275)
    has_validation = False

    def __init__(
        self,
        data_dir: str | Path,
        type: Literal["btcv", "tcia"],
        cropping_csv: str | Path,
    ):
        super().__init__(data_dir, btcv_class_mapping)
        self.type = type
        self.id_to_bbox = self.load_bounding_box(self.type, cropping_csv)

    def load_bounding_box(self, type, path):
        # id, publisher, dataset, original_id, ...
        match_publisher = "Synapse" if type == "btcv" else "TCIA"
        with Path(path).open() as f:
            lines = f.readlines()
            data = [
                line.strip().split(",") for line in lines[1:] if line.strip()
            ]
            return {
                line[3]: [int(_) for _ in line[4:]]
                for line in data if line[1] == match_publisher
            }

    @override
    def load_raw(
        self,
        stage: Literal["train", "val"] = "train"
    ) -> Iterable[tuple[Path, ndarray, ndarray]]:
        if stage == "val":
            return []  # no official validation set
        image_dir = self.data_dir / "image"
        label_dir = self.data_dir / "label"
        if self.type == "btcv":
            for path in image_dir.glob("*.nii.gz"):
                if path.is_file():
                    id = path.name.split(".")[0].removeprefix("img")
                    if id not in self.id_to_bbox:
                        continue
                    image_path = path
                    label_path = label_dir / path.name.replace("img", "label")
                    image, label = self._monai_load(image_path, label_path)
                    yield path, image, label
        else:
            for path in image_dir.glob("PANCREAS_*"):
                if path.is_dir():
                    id = path.name.removeprefix("PANCREAS_")
                    if id not in self.id_to_bbox:
                        continue
                    label_path = label_dir / f"label{id}.nii.gz"
                    zip_path = next(path.rglob("*.zip"))
                    image_path = dicom_zip_to_nii(zip_path)
                    image, label = self._monai_load(image_path, label_path)
                    image_path.unlink()  # remove the temporary nii file
                    yield path, image, label

    def _monai_load(self, image: Path, label: Path):
        d = self.transform({"image": str(image), "label": str(label)})
        return d["image"][0].numpy(), d["label"][0].numpy()  # type: ignore

    @override
    def get_dataset_name(self) -> str:
        return self.type

    @cached_property
    def transform(self):
        return T.Compose([
            T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            T.ScaleIntensityRanged(keys=["image"],
                                   a_min=self.window[0],
                                   a_max=self.window[1],
                                   b_min=0.0,
                                   b_max=1.0,
                                   clip=True),
            T.Orientationd(keys=["image", "label"], axcodes="IPL"),
            T.CropForegroundd(keys=["image", "label"],
                              source_key="label",
                              allow_smaller=True,
                              margin=20),
            T.Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
        ])
