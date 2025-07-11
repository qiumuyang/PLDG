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

from dataset import (Abdomen, AbdomenCT, DGDataset,
                     DistributedMultiDomainSampler, cycle)
from dataset.transform import (AddCutmixBox, DropKeys, FilterLabel,
                               ForegroundCrop, PadTo, RandomFlip, ToTensor)
from evaluate import evaluate
from model.loss import (AggregatedCrossEntropyLoss, AggregatedDiceLoss,
                        BinaryCrossEntropyLoss, BinaryDiceLoss, DiceLoss)
from model.optim import create_lr_scheduler, create_optimizer
from model.prototype import PrototypeBank
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
    accelerator.init_trackers("abdomen-partial-label-dg",
                              config=all_args,
                              init_kwargs=dict(wandb=dict(dir=args.output)))
    return cfg, accelerator


def create_dataset_and_dataloader(args: Namespace, cfg: dict,
                                  accelerator: Accelerator):
    zoo = {"abdomen": Abdomen, "abdomen_ct": AbdomenCT}
    cls = zoo[cfg["dataset"].lower()]

    # check domain classes (tcia does not have class 2 - kidney.R)
    # if cls is AbdomenCT and args.domain != 4:
    #     tcia_id = 3 if args.domain < 4 else 4
    #     if "kidney.R" in cfg["domain_classes"][tcia_id]:
    #         raise ValueError("tcia should not be used with class [kidney.R]")

    # check all domain classes cover all classes
    all_domain_classes = set(sum(cfg["domain_classes"], []))
    if all_domain_classes < set(cfg["class_names"]):
        missing = set(cfg["class_names"]) - all_domain_classes
        raise ValueError(f"missing class {missing} to cover all classes")

    assert cfg["batch_size"] % accelerator.num_processes == 0

    trainset = DGDataset(
        cls,
        root=cfg["root"],
        num_domains=cfg["n_domains"],
        target_domain=args.domain,
        split="train",
        transform=T.Compose([
            FilterLabel(cfg["class_names"],
                        cfg["domain_classes"],
                        domain_key="rel_domain_id",
                        compress=False,
                        add_class_ids=True,
                        keep_original=True),
            ForegroundCrop(*cfg["patch_size"], skip_keys=["classes"]),
            RandomFlip(0.2, axis=-1,
                       keys=("image", "label", "label_original")),
            RandomFlip(0.2, axis=-2,
                       keys=("image", "label", "label_original")),
            RandomFlip(0.2, axis=-3,
                       keys=("image", "label", "label_original")),
            ToTensor(),
            AddCutmixBox(),
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
    trainloader_mix = DataLoader(
        trainset,
        batch_size=cfg["batch_size"] // accelerator.num_processes,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=dataloader_kwargs(
            seed=args.seed * 2,
            rank=accelerator.local_process_index)["worker_init_fn"])
    valloader = DataLoader(valset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True)
    # fix accelerator prepare
    MULTI_DOMAIN_SAMPLER = "multi_domain_sampler"
    setattr(trainloader, MULTI_DOMAIN_SAMPLER, sampler)
    setattr(trainloader_mix, MULTI_DOMAIN_SAMPLER, sampler)
    trainloader, trainloader_mix, valloader = accelerator.prepare(
        trainloader, trainloader_mix, valloader)
    trainloader = cycle(trainloader, MULTI_DOMAIN_SAMPLER)
    trainloader_mix = cycle(trainloader_mix, MULTI_DOMAIN_SAMPLER)
    accelerator.print(f"{trainset}\n{valset}")
    return trainset, valset, trainloader, trainloader_mix, valloader


def get_latest_checkpoint(output: Path) -> tuple[int, Path] | None:
    checkpoints = list(output.joinpath("checkpoints").glob("checkpoint_*"))
    dir_epoch = [(int(c.name.split("_")[-1]), c) for c in checkpoints]
    if not dir_epoch:
        return None
    return max(dir_epoch, key=lambda x: x[0])


def sigmoid_rampup(current: int, rampup_length: int = 200):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    import numpy as np
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def update_foreground_prob(dataset: DGDataset,
                           current: int,
                           rampup_length: int = 200):
    factor = sigmoid_rampup(current, rampup_length)
    if isinstance(dataset.transform, T.Compose):
        for t in dataset.transform.transforms:
            if isinstance(t, ForegroundCrop):
                t.p = (1 - factor) * 0.9
                break


def cal_dice(y_pred, y_true, class_id=None, eps=1e-6):
    if class_id is None:
        return (2 * (y_pred == y_true).sum() /
                (y_true.numel() + y_pred.numel() + eps))
    p = (y_pred == class_id).float()
    q = (y_true == class_id).float()
    return (2 * (p * q).sum() / (p.sum() + q.sum() + eps))


def main():
    args = parse_args()
    cfg, accelerator = setup(args)

    (trainset, _, trainloader, trainloader_mix,
     valloader) = create_dataset_and_dataloader(args, cfg, accelerator)

    max_epoch = min(cfg["epochs"], cfg.get("epoch_stop", float("inf")))
    total_iters = cfg["epochs"] * cfg["iters"]
    actual_iters = max_epoch * cfg["iters"]

    model = VNet(in_channels=cfg["n_channels"],
                 out_channels=cfg["n_classes"],
                 num_filters=cfg["n_filters"],
                 multi_binary=True,
                 lightweight=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = create_optimizer(model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"])
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    total_iters, **cfg.get("lr_kwargs", {}))
    proto_index = cfg["proto_index"]
    proto_bank = PrototypeBank(cfg["n_domains"] - 1,
                               cfg["n_classes"],
                               dim=model.feat_channels[proto_index],
                               mean_updates=cfg["iters"])

    criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    criterion_ce_agg = AggregatedCrossEntropyLoss(n_classes=cfg["n_classes"])
    criterion_dice_agg = AggregatedDiceLoss(n_classes=cfg["n_classes"])
    criterion_bce = BinaryCrossEntropyLoss(n_classes=cfg["n_classes"])
    criterion_bdice = BinaryDiceLoss(n_classes=cfg["n_classes"])
    lambda_dice = cfg.get("lambda_dice", 1)
    model, optimizer, scheduler, proto_bank = accelerator.prepare(
        model, optimizer, scheduler, proto_bank)

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

        # before propagate_epoch, prototypes are estimated from labeled data
        # use labeled-domain classes to initialize unlabeled-domain prototypes
        if epoch == cfg["proto_propagate_epoch"]:
            class_to_domain_ids = {}
            class_names = cfg["class_names"]
            for domain, domain_classes in enumerate(cfg["domain_classes"]):
                for class_name in domain_classes:
                    class_id = class_names.index(class_name)
                    class_to_domain_ids.setdefault(class_id, []).append(domain)
            assert len(
                class_to_domain_ids) == cfg["n_classes"], "Missing class"
            proto_bank.propagate(class_to_domain_ids, skip_background=True)

        for it in range(cfg["iters"]):
            sample, sample_mix = next(trainloader), next(trainloader_mix)
            image, label = sample["image"], sample["label"]
            image2, label2 = sample_mix["image"], sample_mix["label"]
            domain_ids1 = sample["rel_domain_id"]
            domain_ids2 = sample_mix["rel_domain_id"]
            image, image2 = image.float(), image2.float()
            label, label2 = label.long(), label2.long()
            classes, cutmix_box = sample["classes"], sample["cutmix_box"]
            image1, label1 = image.clone(), label.clone()
            _gt1 = sample["label_original"]  # debug only, not used

            with torch.no_grad():
                logits, feats, logits_bin = model(torch.cat([image1, image2]),
                                                  return_binary=True,
                                                  return_feats=True)
                logits1, logits2 = logits.detach().chunk(2)
                logitsb1, logitsb2 = logits_bin.detach().chunk(2)
                feat1, feat2 = feats[proto_index].detach().chunk(2)
                conf1, pseudo1 = logits1.softmax(dim=1).max(dim=1)
                conf2, pseudo2 = logits2.softmax(dim=1).max(dim=1)
                prob1 = logitsb1.sigmoid()
                prob2 = logitsb2.sigmoid()
                _acc_multi = cal_dice(pseudo1, _gt1)
                _pseudo1_bin = prob1.argmax(dim=1) + 1
                _pseudo1_bin[prob1.max(
                    dim=1).values < cfg["non_fg_thresh"]] = 0
                _acc_bin = cal_dice(_pseudo1_bin, _gt1)
                _dice_bg_multi = cal_dice(pseudo1, _gt1, class_id=0)
                _dice_bg_bin = cal_dice(_pseudo1_bin, _gt1, class_id=0)
                # if all binary classifiers considered non-foreground,
                # consider it as background
                non_fg_mask1b = prob1.max(dim=1).values < cfg["non_fg_thresh"]
                non_fg_mask2b = prob2.max(dim=1).values < cfg["non_fg_thresh"]
                pseudo1[non_fg_mask1b] = 0
                pseudo2[non_fg_mask2b] = 0
                # rectify pseudo labels
                if epoch >= cfg["proto_propagate_epoch"]:
                    weight1 = proto_bank.weight_label(
                        feat1,
                        pseudo1,
                        domains=domain_ids1,
                        mode="weighted-across-domain"
                        if epoch < cfg["proto_expand_epoch"] else "in-domain",
                        tau=cfg["proto_temperature"])
                    weight2 = proto_bank.weight_label(
                        feat2,
                        pseudo2,
                        domains=domain_ids2,
                        mode="weighted-across-domain"
                        if epoch < cfg["proto_expand_epoch"] else "in-domain",
                        tau=cfg["proto_temperature"])
                    fore_true_mask = (_gt1 > 0) & (pseudo1 == _gt1)
                    fore_false_mask = (_gt1 > 0) & (pseudo1 != _gt1)
                    back_true_mask = (_gt1 == 0) & (pseudo1 == _gt1)
                    back_false_mask = (_gt1 == 0) & (pseudo1 != _gt1)
                    conf_mask_1 = conf1 > cfg["conf_thresh"]
                    fore_true_weight = weight1[fore_true_mask
                                               & conf_mask_1].mean()
                    fore_false_weight = weight1[fore_false_mask
                                                & conf_mask_1].mean()
                    back_true_weight = weight1[back_true_mask
                                               & conf_mask_1].mean()
                    back_false_weight = weight1[back_false_mask
                                                & conf_mask_1].mean()
                    true_weight = weight1[pseudo1 == _gt1].mean()
                    false_weight = weight1[pseudo1 != _gt1].mean()
                    extra_log_values = dict(ft=fore_true_weight,
                                            ff=fore_false_weight,
                                            bt=back_true_weight,
                                            bf=back_false_weight,
                                            tw=true_weight,
                                            fw=false_weight,
                                            acc_bin=_acc_bin,
                                            acc_multi=_acc_multi,
                                            dice_bg_multi=_dice_bg_multi,
                                            dice_bg_bin=_dice_bg_bin)
                    for k, v in list(extra_log_values.items()):
                        if v.isnan():
                            del extra_log_values[k]
                else:
                    weight1 = torch.zeros_like(pseudo1, dtype=torch.float)
                    weight2 = torch.zeros_like(pseudo2, dtype=torch.float)
                    extra_log_values = dict(acc_bin=_acc_bin,
                                            acc_multi=_acc_multi,
                                            dice_bg_multi=_dice_bg_multi,
                                            dice_bg_bin=_dice_bg_bin)
                # overwrite with ground truth
                pseudo1[label1 > 0] = label1[label1 > 0]
                pseudo2[label2 > 0] = label2[label2 > 0]
                conf1[label1 > 0] = 1
                conf2[label2 > 0] = 1
                weight1[label1 > 0] = 1
                weight2[label2 > 0] = 1
                # quantity & quality metrics from softmatch
                conf_weight1 = (conf1 > cfg["conf_thresh"]).float() * weight1
                correct1 = (_gt1 == pseudo1).float()
                quantity = conf_weight1.mean()
                quality = ((conf_weight1 * correct1).sum() /
                           (conf_weight1.sum() + 1e-6))

                # box: b, d, h, w; image: b, c, d, h, w, pseudo_gt: b, d, h, w
                cutmix_box_im = cutmix_box.unsqueeze(1).expand_as(image1)
                image_mix = image1.clone()
                label_mix = pseudo1.clone()
                conf_mix = conf1.clone()
                weight_mix = weight1.clone()
                image_mix[cutmix_box_im > 0] = image2[cutmix_box_im > 0]
                label_mix[cutmix_box > 0] = pseudo2[cutmix_box > 0]
                conf_mix[cutmix_box > 0] = conf2[cutmix_box > 0]
                weight_mix[cutmix_box > 0] = weight2[cutmix_box > 0]
                conf_mask = (conf_mix > cfg["conf_thresh"]).float()
                conf_weight = weight_mix * conf_mask

            with accelerator.accumulate(model):
                logits, feats, logits_bin = model(torch.cat([image,
                                                             image_mix]),
                                                  return_binary=True,
                                                  return_feats=True)
                pred, pred_mix = logits.chunk(2)
                pred_bin, _ = logits_bin.chunk(2)
                loss_ce = criterion_ce_agg(pred, label, classes)
                loss_dice = criterion_dice_agg(pred, label, classes)
                loss_ce_bin = criterion_bce(pred_bin, label, classes)
                loss_dice_bin = criterion_bdice(pred_bin, label, classes)
                if str(cfg["proto_thresh"]).endswith("%"):
                    proto_rate = float(cfg["proto_thresh"][:-1]) / 100
                    proto_thresh = torch.quantile(conf_weight,
                                                  1 - proto_rate,
                                                  interpolation="nearest")
                else:
                    proto_thresh = cfg["proto_thresh"]
                if conf_weight.sum() > 0:
                    loss_ce_mix = (criterion_ce(pred_mix, label_mix) *
                                   conf_weight).sum() / conf_weight.sum()
                    loss_dice_mix = criterion_dice(pred_mix,
                                                   label_mix,
                                                   ignore=conf_weight
                                                   < proto_thresh)
                else:
                    loss_ce_mix = torch.zeros_like(loss_ce)
                    loss_dice_mix = torch.zeros_like(loss_dice)
                loss_raw = (loss_ce + loss_dice * lambda_dice) / (1 +
                                                                  lambda_dice)
                loss_bin = (loss_ce_bin +
                            loss_dice_bin * lambda_dice) / (1 + lambda_dice)
                loss_mix = (loss_ce_mix +
                            loss_dice_mix * lambda_dice) / (1 + lambda_dice)

                if epoch >= cfg["proto_propagate_epoch"]:
                    loss_contrast = proto_bank.compute_contrastive_loss(
                        feats[proto_index].chunk(2)[0],  # image only
                        pseudo1,
                        domain_ids1,
                        mode="in-domain" if epoch < cfg["proto_expand_epoch"]
                        else "across-domain",
                        ignores=conf1 < cfg["conf_thresh"],
                        skip_background=True,
                    )
                else:
                    loss_contrast = torch.zeros_like(loss_raw)

                if rampup := cfg.get("epoch_rampup"):
                    rampup_iters = rampup * cfg["iters"]
                else:
                    rampup_iters = total_iters
                weight_mix = sigmoid_rampup(epoch * cfg["iters"] + it,
                                            rampup_iters)
                loss = (loss_raw + loss_mix * weight_mix +
                        loss_bin * cfg.get("lambda_bin", 1.0) +
                        loss_contrast * cfg.get("lambda_contrast", 0.1))
                update_foreground_prob(trainset, epoch * cfg["iters"] + it,
                                       rampup_iters)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if cfg["lr_scheduler"] != "plateau":
                    scheduler.step()

                # update prototype bank
                if epoch >= cfg["proto_propagate_epoch"]:
                    # use pseudo label since it is more comprehensive
                    proto_bank.update(feat1.detach(),
                                      pseudo1,
                                      domain_ids1,
                                      ignores=conf1 < cfg["conf_thresh"])
                    proto_bank.update(feat2.detach(),
                                      pseudo2,
                                      domain_ids2,
                                      ignores=conf2 < cfg["conf_thresh"])
                else:
                    proto_bank.update(
                        feats[proto_index].chunk(2)[0].detach(),
                        label,
                        domain_ids1,
                    )

            running_loss += loss.item()
            accelerator.log(
                values=dict(loss=loss.item(),
                            loss_ce=loss_ce.item(),
                            loss_dice=loss_dice.item(),
                            loss_ulb=loss_mix.item(),
                            loss_bin=loss_bin.item(),
                            loss_contrast=loss_contrast.item(),
                            lr=scheduler.get_last_lr()[0],
                            weight_mix=weight_mix,
                            mask_ratio=conf_mask.mean().item(),
                            proto_thresh=proto_thresh,
                            quantity=quantity,
                            quality=quality,
                            **extra_log_values,
                            eta=eta(pbar) + eta_eval(epoch, cfg, eval_time)),
                log_kwargs=dict(wandb=dict(
                    commit=not is_evaluation_epoch or it + 1 != cfg["iters"])),
            )
            pbar.update(1)
            if (it + 1) % log_div == 0:
                pbar.set_postfix(loss=running_loss / (it + 1))

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
                if "save_interval" in cfg and (epoch +
                                               1) % cfg["save_interval"] == 0:
                    accelerator.save_model(model,
                                           args.output / f"model/{epoch:03d}")
                accelerator.save_model(proto_bank,
                                       args.output / f"proto/{epoch:03d}")
        eval_time = timer.get_elapsed_hours()

        accelerator.wait_for_everyone()
        accelerator.save_state()
        accelerator.save_model(model, args.output / "model/latest")

    pbar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
