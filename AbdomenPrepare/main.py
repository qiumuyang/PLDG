from pathlib import Path
from typing import Literal

from process.base import Processor
from process.impl import AMOSProcessor, BTCVProcessor, WORDProcessor
from process.meta import partial_class, shared_classes, shared_palette
from process.utils import dump_as_h5, dump_as_nii, find_axial_bound
from process.visualize import image_label_slices, image_slices, label_slices


def visualize(
    data: Processor,
    modes: list[Literal["image", "image_label", "label"]] = ["image"],
    axis: Literal["axial", "coronal", "sagittal"] = "axial",
    partial_classes: list[str] = [],
    max_samples: int = -1,
    alpha: int = 64,
):
    base = "visualize-partial" if partial_classes else "visualize-setting"
    name = data.get_dataset_name()
    outputs = [Path(f"{base}/{name}/{mode}") for mode in modes]
    [output.mkdir(parents=True, exist_ok=True) for output in outputs]

    palette = shared_palette
    if partial_classes:
        label_remap = partial_class(shared_classes, *partial_classes)
    else:
        label_remap = None
    slicers = []
    for mode in modes:
        match mode:
            case "image":
                slicer = lambda image, label: image_slices(image, axis)
            case "image_label":
                slicer = lambda image, label: image_label_slices(
                    image, label, axis, palette, alpha=alpha)
            case "label":
                slicer = lambda image, label: label_slices(
                    label, axis, palette)
            case _:
                raise ValueError("Invalid mode")
        slicers.append(slicer)

    for sample_id, (path, image, label) in enumerate(data.load()):
        if label_remap is not None:
            label = label_remap[label]
            inf, sup = find_axial_bound(label)
            image = image[inf:sup]
            label = label[inf:sup]
        stem = path.name.split(".")[0]
        for slicer, output in zip(slicers, outputs):
            for i, slice in enumerate(slicer(image, label)):
                vis_path = output / f"{stem}_slice_{i:03d}.png"
                slice.save(vis_path)
        if max_samples >= 0 and (sample_id + 1) >= max_samples:
            break


def dump(data: Processor, fmt: Literal["nii", "h5"] = "nii"):
    name = data.get_dataset_name()

    def save_stage(stage: Literal["train", "val"]):
        output = Path(f"data/processed-{fmt}/{name}/{stage}")
        output.mkdir(parents=True, exist_ok=True)

        for path, image, label in data.load(stage=stage):
            stem = path.name.split(".")[0]
            match fmt:
                case "nii":
                    image_path = output / f"{stem}_image.nii.gz"
                    label_path = output / f"{stem}_label.nii.gz"
                    dump_as_nii(image, image_path)
                    dump_as_nii(label, label_path)
                case "h5":
                    image_label_path = output / f"{stem}.h5"
                    dump_as_h5(image, label, image_label_path)

    save_stage("train")
    if data.has_validation:
        save_stage("val")


def main():
    btcv = BTCVProcessor("data/raw/btcv", "btcv", "cropping.csv")
    tcia = BTCVProcessor("data/raw/tcia", "tcia", "cropping.csv")
    word = WORDProcessor("data/raw/word")
    amos_ = AMOSProcessor.all_sub_processors(
        "data/raw/amos", "labeled_data_meta_0000_0599.csv")
    processors = amos_[:3] + [btcv, tcia, word]

    for processor in processors:
        dump(processor, fmt="h5")


if __name__ == "__main__":
    main()
