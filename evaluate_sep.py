from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm_

from dataset.transform import PadTo
from model.loss import DiceLoss


def evaluate(
    accelerator: Accelerator,
    models: list[nn.Module],
    model_classes: list[list[int]],
    dataloader: DataLoader,
    cfg: dict,
    output_dir: Path | None = None,
    tqdm: dict | None = None,
    allow_overlap: bool = False,
    use_local: bool = True,
) -> list[float]:
    [model.eval() for model in models]

    if tqdm is not None:
        wrap = tqdm_(dataloader,
                     disable=not accelerator.is_local_main_process,
                     dynamic_ncols=True,
                     **tqdm)
    else:
        wrap = dataloader

    dice_classes = np.zeros(cfg["n_classes"])
    samples_seen = 0
    for i, sample in enumerate(wrap):
        image, label = sample["image"].float(), sample["label"].long()
        padding = sample.get("image_pad", None)
        assert image.shape[0] == 1, "Evaluation only supports batch size 1"
        assert padding is not None, "add_pad_info must be True in transform"

        if output_dir is not None and use_local and (output_dir /
                                                     f"{i:04d}.npy").exists():
            pred = torch.from_numpy(np.load(output_dir / f"{i:04d}.npy")).to(
                image.device)
            new_pred = False
        else:
            preds = [
                predict(model,
                        image,
                        padding,
                        classes=model_classes[j],
                        patch_size=cfg["patch_size"],
                        stride_xy=cfg["stride_xy"],
                        stride_z=cfg["stride_z"],
                        num_classes=cfg["n_classes"],
                        batch_size=cfg["batch_size"])
                for j, model in enumerate(models)
            ]
            pred = sum(preds)
            pred[:, 0] = reduce(lambda x, y: x * y, preds, 1)[:, 0]
            new_pred = True

        if output_dir is not None and new_pred:
            out = output_dir / f"{i:04d}.npy"
            np.save(out, pred.cpu().numpy())

        if allow_overlap:
            dc = DiceLoss.dc_overlap(pred, label, cfg["n_classes"]).detach()
        else:
            dc = DiceLoss.dc(pred.argmax(1), label, cfg["n_classes"]).detach()
        dc = accelerator.gather_for_metrics(dc).cpu().numpy()  # type: ignore
        dice_classes += dc.sum(axis=0)
        samples_seen += dc.shape[0]
    dice_classes /= samples_seen
    return dice_classes.tolist()[1:]  # exclude background


@torch.no_grad()
def predict(
    model,
    image: torch.Tensor,
    padding: torch.Tensor,
    classes: list[int],
    patch_size: tuple[int, int, int],
    num_classes: int,
    stride_xy: int = 16,
    stride_z: int = 16,
    batch_size: int = 1,
):
    b, _, d, h, w = image.shape
    pred_sum = torch.zeros((b, num_classes, d, h, w), device=image.device)
    pred_cnt = torch.zeros((b, 1, d, h, w), device=image.device)
    predicted_coords = set()
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        inputs = torch.cat([b[0] for b in buffer], dim=0)
        coords = [b[1] for b in buffer]
        pred = model(inputs)
        pred = pred.softmax(dim=1)
        for p, (z, y, x) in zip(pred, coords):
            # yapf: disable
            pred_sum[...,
                     z:z + patch_size[0],
                     y:y + patch_size[1],
                     x:x + patch_size[2]] += p
            pred_cnt[...,
                     z:z + patch_size[0],
                     y:y + patch_size[1],
                     x:x + patch_size[2]] += 1
            # yapf: enable
        buffer.clear()

    for z in range(0, d, stride_z):
        z = min(z, d - patch_size[0])  # ensure always full patch
        for y in range(0, h, stride_xy):
            y = min(y, h - patch_size[1])
            for x in range(0, w, stride_xy):
                x = min(x, w - patch_size[2])
                if (z, y, x) in predicted_coords:
                    continue
                predicted_coords.add((z, y, x))
                # yapf: disable
                patch = image[...,
                              z:z + patch_size[0],
                              y:y + patch_size[1],
                              x:x + patch_size[2]]
                # yapf: enable
                buffer.append((patch, (z, y, x)))
                if len(buffer) >= batch_size:
                    flush_buffer()
    if buffer:
        flush_buffer()

    pred = pred_sum / pred_cnt
    pred = PadTo.restore_padding(pred, padding)[0]
    foreground_sum = pred[:, classes, ...].sum(dim=1)
    pred[:, 0] = 1 - foreground_sum
    return pred


def main():
    import torchvision.transforms as T
    import yaml
    from safetensors.torch import load_file

    from dataset import Abdomen, AbdomenCT, DGDataset, ToTensor
    from model.vnet import VNet
    from utils import parse_eval_args
    args = parse_eval_args(extra={
        "target-domain": int,
        "type": str,
        "overlap": bool
    })
    assert args.target_domain is not None
    assert args.type in {"latest", "best"}

    cfg = {}
    for config_file in args.config:
        cfg.update(yaml.safe_load(config_file.read_text()))
    accelerator = Accelerator()
    if accelerator.num_processes > 1:
        raise RuntimeError("Evaluation does not support multiprocessing")

    zoo = {"abdomen": Abdomen, "abdomen_ct": AbdomenCT}
    cls = zoo[cfg["dataset"].lower()]

    subdir = []
    for domain in range(cfg["n_domains"]):
        if domain == args.target_domain:
            continue
        if domain < args.target_domain:
            task_id = domain
        else:
            task_id = domain - 1
        domain_alpha = chr(ord("A") + domain)
        task_dir = args.checkpoint / f"{domain_alpha}/{domain}-{task_id}"
        assert task_dir.is_dir(), f"Task {task_dir} does not exist"
        subdir.append(task_dir)

    model_classes = []
    for i in range(cfg["n_domains"] - 1):
        model_classes.append([
            idx for cls in cfg["domain_classes"][i]
            if (idx := cfg["class_names"].index(cls)) > 0
        ])

    valsets = [
        DGDataset(
            cls,
            root=cfg["root"],
            num_domains=cfg["n_domains"],
            target_domain=domain,
            split="val" if args.split == "val" else "traintarget",
            transform=T.Compose([
                # add pad info for restoration
                PadTo(*cfg["patch_size"],
                      add_pad_info=True,
                      skip_keys=["label"]),
                ToTensor(),
            ]),
            use_shm_cache=False,
        ) for domain in args.domain
    ]
    valloaders = [
        DataLoader(valset,
                   batch_size=1,
                   shuffle=False,
                   num_workers=2,
                   pin_memory=True) for valset in valsets
    ]
    if len(valloaders) == 1:
        valloaders = [accelerator.prepare(valloaders[0])]
    else:
        valloaders = accelerator.prepare(*valloaders)

    models = []
    for task_dir in subdir:
        model = VNet(in_channels=cfg["n_channels"],
                     out_channels=cfg["n_classes"],
                     num_filters=cfg["n_filters"])
        model = accelerator.prepare(model)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(
            load_file(task_dir / f"model/{args.type}/model.safetensors"))
        models.append(model)

    if args.overlap:
        dirname = f"{args.target_domain}-preds-overlap"
    else:
        dirname = f"{args.target_domain}-preds"
    output_dir = Path(args.checkpoint).parent / dirname
    output_dir.mkdir(exist_ok=True)

    dice_classes_domains = {}
    for domain, loader in zip(args.domain, valloaders):
        dice_classes = evaluate(accelerator,
                                models,
                                model_classes,
                                loader,
                                cfg,
                                output_dir=output_dir,
                                tqdm=dict(desc=f"Eval Domain {domain}"))
        dice_classes_domains[domain] = dice_classes

    header = ["domain"] + cfg["class_names"][1:]
    print(",".join(header))
    for domain, dice_classes in sorted(dice_classes_domains.items(),
                                       key=lambda x: x[0]):
        row = [str(domain)] + [f"{x:.6f}" for x in dice_classes]
        print(",".join(row))


if __name__ == "__main__":
    main()
