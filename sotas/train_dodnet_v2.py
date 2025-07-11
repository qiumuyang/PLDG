import os
from argparse import Namespace
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, BrightnessTransform,
    ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.noise_transforms import (GaussianBlurTransform,
                                                         GaussianNoiseTransform
                                                         )
from batchgenerators.transforms.resample_transforms import \
    SimulateLowResolutionTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (Abdomen, AbdomenCT, DGDataset,
                     DistributedMultiDomainSampler, DropKeys, FilterLabel,
                     ForegroundCrop, PadTo, RandomFlip, ToTensor, cycle)
from dataset.transform import Compose, MakeCopy
from evaluate import evaluate
from model.optim import create_lr_scheduler, create_optimizer
from sotas.model.dodnet import loss
from sotas.model.dodnet.unet3d import UNet3D, unet3D
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

    assert cfg["batch_size"] % accelerator.num_processes == 0

    trainset = DGDataset(
        cls,
        root=cfg["root"],
        num_domains=cfg["n_domains"],
        target_domain=args.domain,
        split="train",
        transform=Compose([
            FilterLabel(cfg["class_names"],
                        cfg["domain_classes"],
                        domain_key="rel_domain_id",
                        compress=False,
                        add_class_ids=True),
            ForegroundCrop(*cfg["patch_size"], skip_keys=["classes"], p=1.0),
            RandomFlip(0.2, axis=-1, keys=("image", "label")),
            RandomFlip(0.2, axis=-2, keys=("image", "label")),
            RandomFlip(0.2, axis=-3, keys=("image", "label")),
            MakeCopy(["image"]),
            GaussianNoiseTransform(p_per_sample=0.1, data_key="image"),
            GaussianBlurTransform(blur_sigma=(0.5, 1.),
                                  different_sigma_per_channel=True,
                                  p_per_channel=0.5,
                                  p_per_sample=0.2,
                                  data_key="image"),
            BrightnessMultiplicativeTransform((0.75, 1.25),
                                              p_per_sample=0.15,
                                              data_key="image"),
            BrightnessTransform(0.0,
                                0.1,
                                True,
                                p_per_sample=0.15,
                                p_per_channel=0.5,
                                data_key="image"),
            ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"),
            SimulateLowResolutionTransform(zoom_range=(0.5, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.25,
                                           ignore_axes=None,
                                           data_key="image"),
            GammaTransform(gamma_range=(0.7, 1.5),
                           invert_image=False,
                           per_channel=True,
                           retain_stats=True,
                           p_per_sample=0.15,
                           data_key="image"),
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
        transform=Compose([
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

    def __init__(self,
                 model: unet3D,
                 num_classes: int,
                 chunk: int = 3,
                 thresh: float = 0.75):
        self.model = model
        self.num_classes = num_classes
        self.chunk = chunk
        self.thresh = thresh

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = inputs.shape
        task_ids = torch.arange(1, self.num_classes, device=inputs.device)
        output_full = torch.zeros((b, self.num_classes, d, h, w),
                                  device=inputs.device)
        for chunk_task_ids in task_ids.chunk(self.chunk, dim=0):
            task_ids_rep = chunk_task_ids.repeat(inputs.shape[0])
            inputs_rep = inputs.repeat_interleave(len(chunk_task_ids), dim=0)
            outputs = self.model.forward(inputs_rep, task_ids_rep)
            # group by inputs
            outputs_chunk = outputs.chunk(inputs.shape[0], dim=0)
            for i, output in enumerate(outputs_chunk):
                output_full[i, chunk_task_ids] = output.sigmoid().squeeze(1)
        # handle background
        output_full[:, 0] = output_full[:, 1:].max(dim=1).values < self.thresh
        return output_full


def main():
    args = parse_args()
    cfg, accelerator = setup(args)

    (_, _, trainloader,
     valloader) = create_dataset_and_dataloader(args, cfg, accelerator)

    max_epoch = min(cfg["epochs"], cfg.get("epoch_stop", float("inf")))
    total_iters = cfg["epochs"] * cfg["iters"]
    actual_iters = max_epoch * cfg["iters"]

    model = UNet3D(num_classes=cfg["n_classes"], weight_std=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = create_optimizer(model, cfg["optimizer"], cfg["lr"],
                                 cfg["weight_decay"],
                                 **cfg.get("optim_kwargs", {}))
    scheduler = create_lr_scheduler(optimizer, cfg["lr_scheduler"],
                                    total_iters, **cfg.get("lr_kwargs", {}))

    loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=1)  # organ only
    loss_seg_CE = loss.CELoss4MOTS(num_classes=1, ignore_index=255)

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
        # evaluate last epoch only, too slow
        is_evaluation_epoch = epoch == max_epoch - 1

        for it in range(cfg["iters"]):
            sample = next(trainloader)
            images, labels = sample["image"], sample["label"]
            classes = sample["classes"]  # (B, n_classes)

            # task_ids should be one number for each image
            # if multiple classes are present, split into new images
            im, lb, task = [], [], []
            for image, label, cls in zip(images, labels, classes):
                for task_id in torch.unique(cls):
                    if task_id == 0:
                        continue
                    label_t = label == task_id
                    if label_t.sum() > 0:
                        im.append(image)
                        lb.append(label_t)
                        task.append(task_id)
            images = torch.stack(im).float()
            labels = torch.stack(lb).float().unsqueeze(1)
            task_ids = torch.stack(task).long()

            with accelerator.accumulate(model):
                preds = model(images, task_ids)

                term_seg_Dice = loss_seg_DICE.forward(preds, labels)
                term_seg_BCE = loss_seg_CE.forward(preds, labels)
                term_all = term_seg_Dice * 4 + term_seg_BCE

                optimizer.zero_grad()
                accelerator.backward(term_all)
                optimizer.step()
                if cfg["lr_scheduler"] != "plateau":
                    scheduler.step()

            running_loss += term_all.item()
            accelerator.log(
                values=dict(loss=term_all.item(),
                            loss_ce=term_seg_BCE.item(),
                            loss_dice=term_seg_Dice.item(),
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
                                        Adapter(model, cfg["n_classes"]),
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
