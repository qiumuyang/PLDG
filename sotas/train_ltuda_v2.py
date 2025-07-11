import os
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (Abdomen, AbdomenCT, DGDataset,
                     DistributedMultiDomainSampler, cycle)
from dataset.sampler import DistributedMultiDomainSampler
from dataset.transform import (AddCutmixBox, DropKeys, FilterLabel,
                               ForegroundCrop, PadTo, RandomFlip, ToTensor)
from evaluate import evaluate
from model.optim import create_lr_scheduler, create_optimizer
from sotas.model.ltuda import ModelEMA, VNetProto, loss, loss_proto
from utils import (dataloader_kwargs, eta, get_module_version, initialize_seed,
                   parse_args)


def setup(args: Namespace):
    """Load config, initialize seed, and setup accelerator."""

    cfg = {}
    for config_file in args.config:
        cfg.update(yaml.safe_load(config_file.read_text()))
    assert len(cfg["class_names"]) == cfg["n_classes"]
    assert cfg["lr_scheduler"] != "plateau"

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
                        add_class_ids=True),
            ForegroundCrop(*cfg["patch_size"], skip_keys=["classes"], p=0.5),
            RandomFlip(0.2, axis=-1, keys=("image", "label")),
            RandomFlip(0.2, axis=-2, keys=("image", "label")),
            RandomFlip(0.2, axis=-3, keys=("image", "label")),
            AddCutmixBox(p=1.0, key="cutmix_box1"),
            AddCutmixBox(p=1.0, key="cutmix_box2"),
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
    MULTI_DOMAIN_SAMPLER = "multi_domain_sampler"
    setattr(trainloader, MULTI_DOMAIN_SAMPLER, sampler)
    trainloader, valloader = accelerator.prepare(trainloader, valloader)
    trainloader = cycle(trainloader, MULTI_DOMAIN_SAMPLER)
    accelerator.print(f"{trainset}\n{valset}")
    return trainset, valset, trainloader, valloader


class Adapter:

    def __init__(self, model: VNetProto):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        prob = self.model(inputs).sigmoid()
        max_prob, _ = prob.max(dim=1, keepdim=True)
        # equivalent to class_id = 0 if max_prob < 0.5
        prob_bg = 1 - max_prob
        return torch.cat((prob_bg, prob), dim=1)


def label_onehot(labels: torch.Tensor,
                 num_classes: int,
                 annotated_classes: torch.Tensor | None = None):
    # annotated_classes: [B, C]
    onehot = torch.nn.functional.one_hot(labels.long(),
                                         num_classes=num_classes).permute(
                                             0, 4, 1, 2, 3).float()
    # onehot: [B, C, D, H, W]
    if annotated_classes is not None:
        onehot[annotated_classes <= 0] = -1
    return onehot[:, 1:]  # remove background


def shift_cutmix(
    n: int,
    samples: Sequence[torch.Tensor],
    boxes: Sequence[torch.Tensor],
    n_shifts: int = 1,
) -> list[list[torch.Tensor]]:
    results = []
    indices = [(i + n_shifts) % n for i in range(n)]
    for sample in samples:
        sample_shift = sample[indices].detach()
        mixed = []
        for box in boxes:
            if sample.ndim != box.ndim:
                box = box.unsqueeze(1).expand_as(sample)
            sample_mix = sample.detach()
            sample_mix[box > 0] = sample_shift[box > 0]
            mixed.append(sample_mix)
        results.append(mixed)
    return results


def stage_cda(cfg, accelerator, trainloader, valloader, num_epochs):
    student_model = VNetProto(n_channels=cfg["n_channels"],
                              n_classes=cfg["n_classes"] - 1,
                              num_prototype=cfg["n_prototypes"])
    ema_model = ModelEMA(student_model, cfg["ema_decay"], accelerator.device)
    optimizer = create_optimizer(student_model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"],
                                 **cfg.get("optim_kwargs", {}))
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    num_epochs * cfg["iters"],
                                    **cfg.get("lr_kwargs", {}))
    student_model, ema_model.ema, optimizer, scheduler = accelerator.prepare(
        student_model, ema_model.ema, optimizer, scheduler)
    teacher_model = ema_model.ema

    loss_pBCE = loss.pBCE(num_classes=cfg["n_classes"] - 1).to(
        accelerator.device)
    loss_pDice = loss.pDice(num_classes=cfg["n_classes"] - 1).to(
        accelerator.device)

    pbar = tqdm(total=num_epochs * cfg["iters"],
                desc="CDA Epoch 1",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True)

    for epoch in range(num_epochs):

        pbar.set_description(f"CDA Epoch {epoch + 1}")

        is_evaluation_epoch = (epoch + 1) % 2 == 0

        student_model.train()
        teacher_model.train()

        for _ in range(cfg["iters"]):
            sample = next(trainloader)
            images, labels = sample["image"], sample["label"]
            labels = labels.long()
            classes = sample["classes"]
            n = images.shape[0]
            partial_labels_w = label_onehot(labels, cfg["n_classes"], classes)
            cutmix_box1 = sample["cutmix_box1"]
            cutmix_box2 = sample["cutmix_box2"]

            ((images_s1, images_s2), (partial_labels_s1,
                                      partial_labels_s2)) = shift_cutmix(
                                          n,
                                          [images, partial_labels_w],
                                          [cutmix_box1, cutmix_box2],
                                      )

            # generate pseudo labels
            with torch.no_grad():
                preds_w = teacher_model(images).detach()  # [b, c-1, d, h, w]
                prob_t, max_t = preds_w.sigmoid().max(1)
                max_t = max_t + 1  # [b, d, h, w]: [0, c-1) => [1, c)
                pseudo_labels = torch.where(prob_t < 0.5,
                                            torch.zeros_like(max_t), max_t)
                # i.e. masked_pseudo_labels (get_masked_supervision L82)
                pseudo_labels[labels > 0] = labels[labels > 0]
                # create mixed labels
                ((pseudo_labels_s1, pseudo_labels_s2), ) = shift_cutmix(
                    n,
                    [pseudo_labels],
                    [cutmix_box1, cutmix_box2],
                )
                # i.e. decomp (L87-88)
                pseudo_labels_s1_4ch = label_onehot(pseudo_labels_s1,
                                                    cfg["n_classes"])
                pseudo_labels_s2_4ch = label_onehot(pseudo_labels_s2,
                                                    cfg["n_classes"])

            with accelerator.accumulate(student_model):
                outputs = student_model(
                    torch.cat([images, images_s1, images_s2]))
                outputs_w, outputs_s1, outputs_s2 = outputs.chunk(3)
                loss_w = (loss_pBCE.forward(outputs_w, partial_labels_w) +
                          loss_pDice.forward(outputs_w, partial_labels_w) * 4)
                loss_s1 = (
                    loss_pBCE.forward(outputs_s1, pseudo_labels_s1_4ch) +
                    loss_pBCE.forward(outputs_s1, partial_labels_s1))
                loss_s2 = (
                    loss_pBCE.forward(outputs_s2, pseudo_labels_s2_4ch) +
                    loss_pBCE.forward(outputs_s2, partial_labels_s2))
                loss_total = loss_w + loss_s1 + loss_s2

                ema_model.update(student_model)
                teacher_model = ema_model.ema

                optimizer.zero_grad()
                accelerator.backward(loss_total)
                optimizer.step()
                scheduler.step()

            accelerator.log(values=dict(loss=loss_total,
                                        loss_w=loss_w,
                                        loss_s=(loss_s1 + loss_s2) / 2,
                                        lr=scheduler.get_last_lr()[0]))
            pbar.update(1)

        if is_evaluation_epoch:
            accelerator.wait_for_everyone()
            dice_classes = evaluate(accelerator,
                                    Adapter(student_model),
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

    pbar.close()
    return teacher_model, student_model


def stage_pda(cfg, accelerator, trainloader, valloader, num_epochs,
              model: VNetProto, args):
    ema_model = ModelEMA(model, cfg["ema_decay"], accelerator.device)
    optimizer = create_optimizer(model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"],
                                 **cfg.get("optim_kwargs", {}))
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    num_epochs * cfg["iters"],
                                    **cfg.get("lr_kwargs", {}))
    model, ema_model.ema, optimizer, scheduler = accelerator.prepare(
        model, ema_model.ema, optimizer, scheduler)
    model_ema = ema_model.ema

    loss_seg_CE = loss.pBCE(num_classes=cfg["n_classes"] - 1).to(
        accelerator.device)
    loss_seg_DICE = loss.pDice(num_classes=cfg["n_classes"] - 1).to(
        accelerator.device)
    loss_seg_proto = loss_proto.PixelPrototypeCELoss().to(accelerator.device)

    pbar = tqdm(total=num_epochs * cfg["iters"],
                desc="PDA Epoch 1",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True)
    best_dice = 0

    for epoch in range(num_epochs):

        pbar.set_description(f"PDA Epoch {epoch + 1}")

        model.train()
        model_ema.train()
        is_evaluation_epoch = (epoch + 1) % 2 == 0

        for _ in range(cfg["iters"]):
            sample = next(trainloader)
            images, labels = sample["image"], sample["label"]
            labels = labels.long()
            classes = sample["classes"]
            n = images.shape[0]
            partial_labels_w = label_onehot(labels, cfg["n_classes"], classes)
            cutmix_box1 = sample["cutmix_box1"]
            cutmix_box2 = sample["cutmix_box2"]
            ((images_s1, images_s2), ) = shift_cutmix(
                n,
                [images],
                [cutmix_box1, cutmix_box2],
            )

            # generate pseudo labels
            with torch.no_grad():
                preds_w = model_ema(images).detach()
                prob_t, max_t = preds_w.sigmoid().max(1)
                max_t = max_t + 1
                max_t = torch.where(prob_t < 0.5, torch.zeros_like(max_t),
                                    max_t)
                # L124-128: lb channel remains 0, ulb channel uses max_t
                pseudo_label_t = max_t.clone()
                pseudo_label_t[labels > 0] = 0
                # same as L130-135
                mask_labeled = labels.clone()
                mask_labeled[labels == 0] = -1
                mask_unlabeled = pseudo_label_t.clone()
                mask_unlabeled[labels > 0] = -1  # ignore labeled region
                mask_labeled[mask_unlabeled == 0] = 0
                pseudo_label_t[labels > 0] = labels[labels > 0]

                ((pseudo_label_t_s1, pseudo_label_t_s2), (mask_labeled_s1,
                                                          mask_labeled_s2),
                 (mask_unlabeled_s1, mask_unlabeled_s2)) = shift_cutmix(
                     n,
                     [pseudo_label_t, mask_labeled, mask_unlabeled],
                     [cutmix_box1, cutmix_box2],
                 )

            with accelerator.accumulate(model):
                preds_w_1, preds_s1_linear, preds_s2_linear = model(
                    torch.cat([images, images_s1, images_s2])).chunk(3)
                term_model_1 = (
                    loss_seg_CE.forward(preds_w_1, partial_labels_w) +
                    loss_seg_DICE.forward(preds_w_1, partial_labels_w) * 4)
                (preds_s1_protol, contrast_logits_s1l, contrast_target_s1l,
                 preds_s1_protoul, contrast_logits_s1ul,
                 contrast_target_s1ul) = model(images_s1,
                                               mask_labeled_s1,
                                               mask_unlabeled_s1,
                                               use_prototype=True)
                (preds_s2_protol, contrast_logits_s2l, contrast_target_s2l,
                 preds_s2_protoul, contrast_logits_s2ul,
                 contrast_target_s2ul) = model(images_s2,
                                               mask_labeled_s2,
                                               mask_unlabeled_s2,
                                               use_prototype=True)

                pseudo_label_t_s1_4cha = label_onehot(pseudo_label_t_s1,
                                                      cfg["n_classes"])
                cps_ce_s1_linear = loss_seg_CE.forward(preds_s1_linear,
                                                       pseudo_label_t_s1_4cha)
                cps_ce_s1_protol = loss_seg_proto.forward(
                    preds_s1_protol, contrast_logits_s1l, contrast_target_s1l,
                    pseudo_label_t_s1.long())
                cps_ce_s1_protoul = loss_seg_proto.forward(
                    preds_s1_protoul, contrast_logits_s1ul,
                    contrast_target_s1ul, pseudo_label_t_s1.long())

                pseudo_label_t_s2_4cha = label_onehot(pseudo_label_t_s2,
                                                      cfg["n_classes"])
                cps_ce_s2_linear = loss_seg_CE.forward(preds_s2_linear,
                                                       pseudo_label_t_s2_4cha)
                cps_ce_s2_protol = loss_seg_proto(preds_s2_protol,
                                                  contrast_logits_s2l,
                                                  contrast_target_s2l,
                                                  pseudo_label_t_s2.long())
                cps_ce_s2_protoul = loss_seg_proto(preds_s2_protoul,
                                                   contrast_logits_s2ul,
                                                   contrast_target_s2ul,
                                                   pseudo_label_t_s2.long())

                # yapf: disable
                term_model_2 = (1 * (cps_ce_s1_linear + cps_ce_s2_linear) +
                                1 * (cps_ce_s1_protol + cps_ce_s2_protol) +
                                1 * (cps_ce_s1_protoul + cps_ce_s2_protoul))
                # yapf: enable
                loss_total = term_model_1 + term_model_2

                ema_model.update(model)
                model_ema = ema_model.ema

                optimizer.zero_grad()
                accelerator.backward(loss_total)
                optimizer.step()
                scheduler.step()

            accelerator.log(values=dict(
                loss=loss_total,
                lr=scheduler.get_last_lr()[0],
                eta=eta(pbar),
            ))
            pbar.update(1)

        accelerator.wait_for_everyone()
        if is_evaluation_epoch:
            dice_classes = evaluate(accelerator,
                                    Adapter(model),
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
            if dice > best_dice:
                best_dice = dice
                accelerator.save_model(model, args.output / "model/best")
        accelerator.save_state()
        accelerator.save_model(model, args.output / "model/latest")
        accelerator.save_model(model_ema, args.output / "model/latest-ema")

    pbar.close()


def main():
    args = parse_args()
    cfg, accelerator = setup(args)

    (_, _, trainloader,
     valloader) = create_dataset_and_dataloader(args, cfg, accelerator)

    max_epoch = min(cfg["epochs"], cfg.get("epoch_stop", float("inf")))

    teacher, student = stage_cda(cfg, accelerator, trainloader, valloader,
                                 max_epoch // 2)
    del teacher
    stage_pda(cfg, accelerator, trainloader, valloader, max_epoch // 2,
              student, args)
    accelerator.end_training()


if __name__ == "__main__":
    main()
