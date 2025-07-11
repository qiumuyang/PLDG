from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm_

from dataset.transform import PadTo
from model.loss import DiceLoss


def evaluate(
    accelerator: Accelerator,
    model,
    dataloader: DataLoader,
    cfg: dict,
    output_dir: Path | None = None,
    tqdm: dict | None = None,
    exclude_background: bool = True,
    outputs_prob: bool = False,
    allow_overlap: bool = False,
    use_local: bool = True,
) -> list[float]:
    model.eval()

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
                                                     f"{i:04d}.npz").exists():
            pred = torch.from_numpy(
                np.load(output_dir / f"{i:04d}.npz")["pred"]).to(image.device)
            new_pred = False
        else:
            pred = predict(model,
                           image,
                           padding,
                           patch_size=cfg["patch_size"],
                           stride_xy=cfg["stride_xy"],
                           stride_z=cfg["stride_z"],
                           num_classes=cfg["n_classes"],
                           batch_size=cfg["batch_size"] //
                           accelerator.num_processes,
                           outputs_prob=outputs_prob,
                           allow_overlap=allow_overlap)
            new_pred = True
        if output_dir is not None and new_pred:
            out = output_dir / f"{i:04d}.npz"
            np.savez_compressed(out, pred=pred.cpu().numpy())

        if allow_overlap:
            dc = DiceLoss.dc_overlap(pred, label, cfg["n_classes"]).detach()
        else:
            dc = DiceLoss.dc(pred, label, cfg["n_classes"]).detach()
        dc = accelerator.gather_for_metrics(dc).cpu().numpy()  # type: ignore
        dice_classes += dc.sum(axis=0)
        samples_seen += dc.shape[0]
    dice_classes /= samples_seen
    if exclude_background:
        return dice_classes.tolist()[1:]  # type: ignore
    return dice_classes.tolist()  # type: ignore


@torch.no_grad()
def predict(
    model,
    image: torch.Tensor,
    padding: torch.Tensor,
    patch_size: tuple[int, int, int],
    num_classes: int,
    stride_xy: int = 16,
    stride_z: int = 16,
    batch_size: int = 1,
    outputs_prob: bool = False,
    allow_overlap: bool = False,
):
    """Use a sliding-window strategy to predict the image."""

    b, _, d, h, w = image.shape
    pred_sum = torch.zeros((b, num_classes, d, h, w), device=image.device)
    pred_cnt = torch.zeros((b, 1, d, h, w), device=image.device)
    predicted_coords = set()
    buffer = []

    # since no grad requires less memory
    # larger batch_size can make evaluation faster
    batch_size *= 2

    def flush_buffer():
        if not buffer:
            return
        inputs = torch.cat([b[0] for b in buffer], dim=0)
        coords = [b[1] for b in buffer]
        pred = model(inputs)
        if not outputs_prob:
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
    if not allow_overlap:
        return pred.argmax(dim=1)
    return pred > 0.5


def main():
    import torchvision.transforms as T
    import yaml
    from safetensors.torch import load_file

    from dataset import Abdomen, AbdomenCT, DGDataset, ToTensor
    from model.vnet import VNet
    from utils import parse_eval_args
    args = parse_eval_args(dict(bg=bool, sota=str, overlap=bool))

    cfg = {}
    for config_file in args.config:
        cfg.update(yaml.safe_load(config_file.read_text()))
    accelerator = Accelerator()
    if accelerator.num_processes > 1:
        raise RuntimeError("Evaluation does not support multiprocessing")

    zoo = {"abdomen": Abdomen, "abdomen_ct": AbdomenCT}
    cls = zoo[cfg["dataset"].lower()]

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

    match args.sota:
        case None:
            model = VNet(in_channels=cfg["n_channels"],
                         out_channels=cfg["n_classes"],
                         num_filters=cfg["n_filters"])
        case "dodnet":
            from sotas.model.dodnet.unet3d import UNet3D
            model = UNet3D(num_channels=cfg["n_channels"],
                           num_classes=cfg["n_classes"],
                           weight_std=True)
        case "versatile":
            from sotas.model.versatile import VisionTransformer as ViT_seg
            from sotas.model.versatile import get_vit_3d_config
            vit_seg = get_vit_3d_config()
            model = ViT_seg(vit_seg,
                            img_size=cfg["patch_size"],
                            num_classes=cfg["n_classes"],
                            in_channels=cfg["n_channels"])
        case "ltuda":
            from sotas.model.ltuda import VNetProto
            model = VNetProto(n_channels=cfg["n_channels"],
                              n_classes=cfg["n_classes"] - 1,
                              num_prototype=5)
        case _:
            raise NotImplementedError(args.sota)
    model, *valloaders = accelerator.prepare(model, *valloaders)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(load_file(args.checkpoint), strict=False)

    if args.overlap:
        dirname = "preds-overlap"
    else:
        dirname = "preds"
    output_dir = Path(args.checkpoint).parent / dirname
    output_dir.mkdir(exist_ok=True)

    if args.sota == "dodnet":
        from sotas.train_dodnet import Adapter
        model = Adapter(model, num_classes=cfg["n_classes"], thresh=0.5)
        outputs_prob = True
    elif args.sota == "ltuda":
        from sotas.train_ltuda import Adapter
        model = Adapter(model)
        outputs_prob = True
    else:
        outputs_prob = False

    dice_classes_domains = {}
    for domain, loader in zip(args.domain, valloaders):
        dice_classes = evaluate(accelerator,
                                model,
                                loader,
                                cfg,
                                output_dir=output_dir,
                                tqdm=dict(desc=f"Eval Domain {domain}"),
                                exclude_background=not args.bg,
                                allow_overlap=args.overlap,
                                outputs_prob=outputs_prob)
        dice_classes_domains[domain] = dice_classes

    header = ["domain"] + cfg["class_names"][0 if args.bg else 1:]
    print(",".join(header))
    for domain, dice_classes in sorted(dice_classes_domains.items(),
                                       key=lambda x: x[0]):
        row = [str(domain)] + [f"{x:.6f}" for x in dice_classes]
        print(",".join(row))


if __name__ == "__main__":
    main()
