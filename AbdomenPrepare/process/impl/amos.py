from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

from numpy import ndarray
from typing_extensions import override

import process.transforms as T

from ..base import Processor
from ..meta import amos_class_mapping


class AMOSProcessor(Processor):

    TYPE_TO_MODEL = {
        "ct1": "Aquilion ONE",
        "ct2": "Brilliance16",
        "ct3": "SOMATOM Force",
        "mr": "Prisma/SIGNA HDe",
    }

    margin_d = (10, 10)
    margin_h = (15, 15)
    margin_w = (25, 25)
    window = (-175, 250)
    has_validation = True

    def __init__(
        self,
        data_dir: str | Path,
        type: Literal["ct1", "ct2", "ct3", "mr"],
        metadata_csv: str | Path,
    ):
        super().__init__(data_dir, amos_class_mapping)
        self.type = type
        self.ids = self.load_id_by_metadata(metadata_csv, type)

    @classmethod
    def all_sub_processors(cls, data_dir: str | Path, meta: str | Path):
        return [
            cls(data_dir, "ct1", meta),
            cls(data_dir, "ct2", meta),
            cls(data_dir, "ct3", meta),
            cls(data_dir, "mr", meta),
        ]

    def load_id_by_metadata(self, path: str | Path, type: str) -> set[int]:
        # amos_id,BirthDate,Sex,Age,Model Name,...
        match_models = self.TYPE_TO_MODEL[type].split("/")

        with Path(path).open() as f:
            lines = f.readlines()
            data = [
                line.strip().split(",") for line in lines[1:] if line.strip()
            ]
            return {int(line[0]) for line in data if line[4] in match_models}

    @override
    def get_dataset_name(self) -> str:
        return f"amos_{self.type}"

    @override
    def load_raw(
        self,
        stage: Literal["train", "val"] = "train",
    ) -> Iterable[tuple[Path, ndarray, ndarray]]:
        image_dir = self.data_dir / ("imagesTr"
                                     if stage == "train" else "imagesVa")
        label_dir = self.data_dir / ("labelsTr"
                                     if stage == "train" else "labelsVa")
        for path in sorted(image_dir.glob("*.nii.gz")):
            id = path.name.split(".")[0].removeprefix("amos_")
            if int(id) not in self.ids:
                continue
            label_path = label_dir / path.name
            result = self.transform({
                "image": str(path),
                "label": str(label_path)
            })
            image = result["image"][0].numpy()  # type: ignore
            label = result["label"][0].numpy()  # type: ignore
            yield path, image, label

    @cached_property
    def transform(self):
        if self.type == "mr":
            intensity = T.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        else:
            intensity = T.ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window[0],
                a_max=self.window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        return T.Compose([
            T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            intensity,
            T.CropForegroundd(keys=["image", "label"],
                              source_key="image",
                              allow_smaller=True),
            T.Orientationd(keys=["image", "label"], axcodes="IPL"),
            T.Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
        ])
