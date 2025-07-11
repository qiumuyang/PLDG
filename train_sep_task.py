import os
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (Abdomen, AbdomenCT, DGDataset,
                     DistributedMultiDomainSampler, cycle)
from dataset.transform import (DropKeys, FilterLabel, ForegroundCrop, PadTo,
                               RandomFlip, ToTensor)
from evaluate import evaluate
from model.loss import DiceLoss
from model.optim import create_lr_scheduler, create_optimizer
from model.vnet import VNet
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

    split_classes = cfg["domain_classes"][args.task_id]
    cfg["domain_classes"] = [split_classes]
    class_ids = [cfg["class_names"].index(c) for c in split_classes]
    class_ids = [i for i in class_ids if i >= 0]

    modules = get_module_version()
    all_args = {
        **cfg,
        **vars(args),
        "num_processes": accelerator.num_processes,
        "version": modules,
        "entrypoint": Path(__file__).name,
        "classes": split_classes,
        "hostname": os.getenv("CUSTOM_HOST") or "N/A",
    }
    args.output.mkdir(exist_ok=True, parents=True)
    accelerator.init_trackers("abdomen-partial-label-dg-septask-final",
                              config=all_args,
                              init_kwargs=dict(wandb=dict(dir=args.output)))
    return cfg, accelerator, class_ids


def create_dataset_and_dataloader(args: Namespace, cfg: dict,
                                  accelerator: Accelerator):
    zoo = {"abdomen": Abdomen, "abdomen_ct": AbdomenCT}
    cls = zoo[cfg["dataset"].lower()]

    assert cfg["batch_size"] % accelerator.num_processes == 0

    trainset = DGDataset(
        cls,
        root=cfg["root"],
        num_domains=cfg["n_domains"],
        target_domain=args.domain,
        split="traintarget",
        transform=T.Compose([
            FilterLabel(cfg["class_names"],
                        cfg["domain_classes"],
                        domain_key="rel_domain_id",
                        compress=False),
            ForegroundCrop(*cfg["patch_size"], skip_keys=["classes"]),
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
        split="val-single",
        transform=T.Compose([
            # add pad info for restoration
            PadTo(*cfg["patch_size"], add_pad_info=True, skip_keys=["label"]),
            ToTensor(),
        ]),
    )
    sampler = DistributedMultiDomainSampler.from_dataset(
        trainset,
        shuffle=True,
        balanced=cfg["balanced"],
        seed=args.seed + accelerator.local_process_index)
    trainloader = DataLoader(
        trainset,
        batch_size=cfg["batch_size"] // accelerator.num_processes,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=dataloader_kwargs(
            seed=args.seed,
            rank=accelerator.local_process_index)["worker_init_fn"])
    valloader = DataLoader(valset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=2,
                           pin_memory=True)
    # fix accelerator prepare
    MULTI_DOMAIN_SAMPLER = "multi_domain_sampler"
    setattr(trainloader, MULTI_DOMAIN_SAMPLER, sampler)
    trainloader, valloader = accelerator.prepare(trainloader, valloader)
    trainloader = cycle(trainloader, MULTI_DOMAIN_SAMPLER)
    accelerator.print(f"{trainset}\n{valset}")
    return trainset, valset, trainloader, valloader


def get_latest_checkpoint(output: Path) -> tuple[int, Path] | None:
    checkpoints = list(output.joinpath("checkpoints").glob("checkpoint_*"))
    dir_epoch = [(int(c.name.split("_")[-1]), c) for c in checkpoints]
    if not dir_epoch:
        return None
    return max(dir_epoch, key=lambda x: x[0])


def main():
    args = parse_args(extra={"task-id": int})
    if args.task_id is None:
        raise ValueError("--task-id is required")
    args.output = args.output / f"{args.domain}-{args.task_id}"

    cfg, accelerator, class_ids = setup(args)
    accelerator.print(f"Class IDs: {class_ids}")

    (_, _, trainloader,
     valloader) = create_dataset_and_dataloader(args, cfg, accelerator)

    max_epoch = min(cfg["epochs"], cfg.get("epoch_stop", float("inf")))
    total_iters = cfg["epochs"] * cfg["iters"]
    actual_iters = max_epoch * cfg["iters"]

    model = VNet(in_channels=cfg["n_channels"],
                 out_channels=cfg["n_classes"],
                 num_filters=cfg["n_filters"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = create_optimizer(model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"])
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    total_iters, **cfg.get("lr_kwargs", {}))

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    model, optimizer, scheduler = accelerator.prepare(model, optimizer,
                                                      scheduler)

    if (latest := get_latest_checkpoint(args.output)) is not None:
        start_epoch, checkpoint = latest
        accelerator.load_state(str(checkpoint))
        accelerator.print(f"Loaded checkpoint at epoch {start_epoch}\n"
                          f"Resuming training from epoch {start_epoch + 1}")
        start_epoch += 1
        actual_iters -= start_epoch * cfg["iters"]
    else:
        start_epoch = 0

    pbar = tqdm(total=actual_iters,
                desc="Epoch 1",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True)
    log_div = round(cfg["iters"] / cfg["logs_each_epoch"])
    best_dice = 0
    eval_time = 0

    for epoch in range(start_epoch, max_epoch):

        model.train()
        pbar.set_description(f"Epoch {epoch + 1}")
        running_loss = 0
        is_evaluation_epoch = epoch >= cfg.get("warmup_epochs", -1)

        for it in range(cfg["iters"]):
            sample = next(trainloader)
            image, label = sample["image"], sample["label"]
            image = image.float()
            label = label.long()

            with accelerator.accumulate(model):
                pred = model(image)
                loss_ce = criterion_ce(pred, label)
                loss_dice = criterion_dice(pred, label, softmax="softmax")
                loss = (loss_ce + loss_dice) / 2
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if cfg["lr_scheduler"] != "plateau":
                    scheduler.step()

            running_loss += loss.item()
            accelerator.log(
                values=dict(loss=loss.item(),
                            loss_ce=loss_ce.item(),
                            loss_dice=loss_dice.item(),
                            lr=scheduler.get_last_lr()[0],
                            eta=eta(pbar) + eta_eval(epoch, cfg, eval_time)),
                log_kwargs=dict(wandb=dict(
                    commit=not is_evaluation_epoch or it + 1 != cfg["iters"])),
            )
            pbar.update(1)
            if (it + 1) % log_div == 0:
                pbar.set_postfix(loss=running_loss / (it + 1))

        if cfg["lr_scheduler"] == "plateau":
            scheduler.step(running_loss / cfg["iters"])

        # ====> end of epoch
        with Timer() as timer:
            if is_evaluation_epoch:
                accelerator.wait_for_everyone()
                dice_classes = evaluate(accelerator,
                                        model,
                                        valloader,
                                        cfg,
                                        tqdm=dict(position=1, leave=False))
                # take the class of interest
                dice_interest = []
                for class_id in class_ids:
                    # -1 due to background removed
                    dice_interest.append(dice_classes[class_id - 1])
                dice_classes = dice_interest
                dice = sum(dice_classes) / len(dice_classes)
                accelerator.log(
                    dict(dice=dice,
                         **{
                             f"dice_{cfg['class_names'][cid]}": dc
                             for cid, dc in zip(class_ids, dice_classes)
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
