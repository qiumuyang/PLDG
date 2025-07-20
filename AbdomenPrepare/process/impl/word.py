from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

from numpy import ndarray
from typing_extensions import override

import process.transforms as T

from ..base import Processor
from ..meta import word_class_mapping


class WORDProcessor(Processor):

    margin_d = (10, 10)
    margin_h = (10, 10)
    margin_w = (20, 20)
    window = (-125, 275)
    has_validation = True

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir, word_class_mapping)

    @override
    def load_raw(
        self,
        stage: Literal["train", "val"] = "train"
    ) -> Iterable[tuple[Path, ndarray, ndarray]]:
        image_dir = self.data_dir / ("imagesTr"
                                     if stage == "train" else "imagesVal")
        label_dir = self.data_dir / ("labelsTr"
                                     if stage == "train" else "labelsVal")
        for path in image_dir.glob("*.nii.gz"):
            if path.is_file():
                label_path = label_dir / path.name

                # image = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
                # label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
                result = self.transform({
                    "image": str(path),
                    "label": str(label_path)
                })
                image = result["image"][0].numpy()  # type: ignore
                label = result["label"][0].numpy()  # type: ignore
                yield path, image, label

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
