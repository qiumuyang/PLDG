import os
from argparse import Namespace
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (Abdomen, AbdomenCT, DGDataset, DropKeys, FairDGDataset,
                     FilterLabel, ForegroundCrop, PadTo, RandomFlip, ToTensor,
                     cycle)
from evaluate import evaluate
from model.optim import create_lr_scheduler, create_optimizer
from sotas.model.versatile import BalancedCELoss, DiceLoss
from sotas.model.versatile import VisionTransformer as ViT_seg
from sotas.model.versatile import get_vit_3d_config
from utils import (Timer, dataloader_kwargs, eta, eta_eval, get_module_version,
                   initialize_seed, parse_args)


def setup(args: Namespace):
    """Load config, initialize seed, and setup accelerator."""

    cfg = {}
    for config_file in args.config:
        cfg.update(yaml.safe_load(config_file.read_text()))
    assert len(cfg["class_names"]) == cfg["n_classes"]

    if "num_threads" in cfg:
        torch.set_num_threads(cfg["num_threads"])
    cfg["num_workers"] = min(cfg["num_workers"],
                             os.cpu_count())  # type: ignore

    proj = ProjectConfiguration(project_dir=args.output,
                                automatic_checkpoint_naming=True,
                                total_limit=1)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["mixed_precision"],
        step_scheduler_with_optimizer=False,  # True will cause extra steps
        log_with="wandb",
        project_config=proj,
    )

    initialize_seed(args.seed, rank=accelerator.local_process_index)

    modules = get_module_version()
    all_args = {
        **cfg,
        **vars(args),
        "num_processes": accelerator.num_processes,
        "version": modules,
        "entrypoint": Path(__file__).name,
        "hostname": os.getenv("CUSTOM_HOST") or "N/A",
    }
    args.output.mkdir(exist_ok=True, parents=True)
    accelerator.init_trackers("abdomen-partial-label-dg-sotas",
                              config=all_args,
                              init_kwargs=dict(wandb=dict(dir=args.output)))
    return cfg, accelerator


def create_dataset_and_dataloader(args: Namespace, cfg: dict,
                                  accelerator: Accelerator):
    zoo = {"abdomen": Abdomen, "abdomen_ct": AbdomenCT}
    cls = zoo[cfg["dataset"].lower()]

    # check all domain classes cover all classes
    all_domain_classes = set(sum(cfg["domain_classes"], []))
    if all_domain_classes < set(cfg["class_names"]):
        missing = set(cfg["class_names"]) - all_domain_classes
        raise ValueError(f"missing class {missing} to cover all classes")

    class_to_domain: dict[int, list[int]] = {}
    for domain_id, classes in enumerate(cfg["domain_classes"]):
        for class_name in classes:
            class_id = cfg["class_names"].index(class_name)
            class_to_domain.setdefault(class_id, []).append(domain_id)
    class_to_domain.pop(0, None)  # remove background class

    trainset = FairDGDataset(
        cls,
        root=cfg["root"],
        num_domains=cfg["n_domains"],
        target_domain=args.domain,
        split="train",
        class_to_domain=class_to_domain,
        transform=T.Compose([
            FilterLabel(cfg["class_names"],
                        cfg["domain_classes"],
                        domain_key="rel_domain_id",
                        compress=False,
                        add_class_ids=True),
            ForegroundCrop(*cfg["patch_size"], skip_keys=["classes"], p=0.9),
            RandomFlip(0.2, axis=-1, keys=("image", "label")),
            RandomFlip(0.2, axis=-2, keys=("image", "label")),
            RandomFlip(0.2, axis=-3, keys=("image", "label")),
            ToTensor(),
            DropKeys("class_ids"),
        ]),
    )
    valset = DGDataset(
        cls,
        root=cfg["root"],
        num_domains=cfg["n_domains"],
        target_domain=args.domain,
        split="val",
        transform=T.Compose([
            # add pad info for restoration
            PadTo(*cfg["patch_size"], add_pad_info=True, skip_keys=["label"]),
            ToTensor(),
        ]),
        use_shm_cache=False,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=dataloader_kwargs(
            seed=args.seed,
            rank=accelerator.local_process_index)["worker_init_fn"])
    valloader = DataLoader(valset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True)
    trainloader, valloader = accelerator.prepare(trainloader, valloader)
    trainloader = cycle(trainloader)
    accelerator.print(f"{trainset}\n{valset}")
    return trainset, valset, trainloader, valloader


def main():
    args = parse_args()
    cfg, accelerator = setup(args)

    (_, _, trainloader,
     valloader) = create_dataset_and_dataloader(args, cfg, accelerator)

    max_epoch = min(cfg["epochs"], cfg.get("epoch_stop", float("inf")))
    total_iters = cfg["epochs"] * cfg["iters"]
    actual_iters = max_epoch * cfg["iters"]

    vit_seg = get_vit_3d_config()
    model = ViT_seg(vit_seg,
                    img_size=cfg["patch_size"],
                    num_classes=cfg["n_classes"],
                    in_channels=cfg["n_channels"])
    optimizer = create_optimizer(model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"],
                                 **cfg.get("optim_kwargs", {}))
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    total_iters, **cfg.get("lr_kwargs", {}))

    loss_seg_DICE = DiceLoss(n_classes=cfg["n_classes"])
    loss_seg_CE = BalancedCELoss(
        n_classes=cfg["n_classes"],
        beta=1.0,
        gamma=2.0,
        multiplier_for_unlabeled_data=cfg["weight_entropy_minimization"])

    model, optimizer, scheduler = accelerator.prepare(model, optimizer,
                                                      scheduler)

    pbar = tqdm(total=actual_iters,
                desc="Epoch 1",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True)
    best_dice = 0
    eval_time = 0

    for epoch in range(max_epoch):

        model.train()
        pbar.set_description(f"Epoch {epoch + 1}")
        running_loss = 0
        is_evaluation_epoch = (epoch + 1) % 2 == 0

        for it in range(cfg["iters"]):
            sample = next(trainloader)
            images, labels = sample["image"], sample["label"]
            annotated_categories = sample["classes"]
            images = images.float()
            labels = labels.long()

            with accelerator.accumulate(model):
                preds = model(images)

                max_along_axis = torch.max(preds, dim=1, keepdim=True).values
                exp_logits = torch.exp(preds - max_along_axis)
                probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

                loss_wce, regularizer = loss_seg_CE(probs, labels,
                                                    annotated_categories)
                loss_dice = loss_seg_DICE(probs, labels, annotated_categories)

                loss = 1.0 * loss_wce + 1.0 * loss_dice + 1.0 * regularizer

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if cfg["lr_scheduler"] != "plateau":
                    scheduler.step()

            running_loss += loss.item()
            accelerator.log(
                values=dict(loss=loss.item(),
                            loss_ce=loss_wce.item(),
                            loss_dice=loss_dice.item(),
                            loss_ulb=regularizer.item(),
                            lr=scheduler.get_last_lr()[0],
                            eta=eta(pbar) + eta_eval(epoch, cfg, eval_time)),
                log_kwargs=dict(wandb=dict(
                    commit=not is_evaluation_epoch or it + 1 != cfg["iters"])),
            )
            pbar.update(1)

        # ====> end of epoch
        if cfg["lr_scheduler"] == "plateau":
            scheduler.step(running_loss / cfg["iters"])

        with Timer() as timer:
            if is_evaluation_epoch:
                accelerator.wait_for_everyone()
                dice_classes = evaluate(accelerator,
                                        model,
                                        valloader,
                                        cfg,
                                        tqdm=dict(position=1, leave=False))
                dice = sum(dice_classes) / len(dice_classes)
                accelerator.log(
                    dict(dice=dice,
                         **{
                             f"dice_{cfg['class_names'][i]}": d
                             for i, d in enumerate(dice_classes, 1)
                         }))
                if accelerator.is_local_main_process:
                    tqdm.write(f"Epoch {epoch + 1}: dice={dice * 100:.2f}%")
                if dice > best_dice:
                    best_dice = dice
                    accelerator.save_model(model, args.output / "model/best")
        eval_time = timer.get_elapsed_hours()

        accelerator.wait_for_everyone()
        accelerator.save_state()
        accelerator.save_model(model, args.output / "model/latest")

    pbar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
