import torch.optim as optim


def create_optimizer(model, name: str, lr: float, weight_decay: float,
                     **kwargs):
    match name.lower():
        case "adamw":
            return optim.AdamW(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay,
                               **kwargs)
        case "sgd":
            return optim.SGD(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay,
                             **kwargs)
        case "adam":
            return optim.Adam(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay,
                              **kwargs)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")


class OverrideLRScheduler(optim.lr_scheduler.LRScheduler):
    """
    An lr scheduler that overrides the learning rate of another.

    Attributes:
        scheduler: The scheduler to override.
        override_from: The start step to override.
        override_lr: The learning rate to override.
    """

    def __init__(self, scheduler: optim.lr_scheduler.LRScheduler,
                 override_from: int, override_lr: float):
        self.scheduler = scheduler
        self.override_from = override_from
        self.override_lr = override_lr
        self.optimizer = scheduler.optimizer
        self.last_epoch = scheduler.last_epoch

    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
        self.last_epoch = self.scheduler.last_epoch
        if self.last_epoch >= self.override_from:
            for group in self.optimizer.param_groups:
                group["lr"] = self.override_lr

    def get_lr(self):
        if self.last_epoch < self.override_from:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            self.override_lr for _ in range(len(self.optimizer.param_groups))
        ]

    def get_last_lr(self):
        if self.last_epoch <= self.override_from:
            return self.scheduler.get_last_lr()
        return [
            self.override_lr for _ in range(len(self.optimizer.param_groups))
        ]

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


def create_lr_scheduler(optimizer, strategy: str, total_iters: int,
                        **kwargs) -> optim.lr_scheduler.LRScheduler:
    override_from = kwargs.pop("override_from", None)
    override_lr = kwargs.pop("override_lr", None)
    match strategy.lower():
        case "cosine":
            base = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)
        case "poly":
            base = optim.lr_scheduler.LambdaLR(
                optimizer, lambda it: (1 - it / total_iters)**0.9)
        case "none":
            base = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        case "plateau":
            base = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        case _:
            raise ValueError(f"Unknown lr scheduler: {strategy}")
    if override_from is not None and override_lr is not None:
        return OverrideLRScheduler(base, override_from, override_lr)
    return base
