import importlib
import random
from typing import Callable, TypedDict

import numpy as np
import torch


class DataLoaderRandomKw(TypedDict):
    worker_init_fn: Callable[[int], None]
    generator: torch.Generator


MODULES = [
    "accelerate",
    "numpy",
    "PIL",
    "torch",
    "torchvision",
]


# https://pytorch.org/docs/stable/notes/randomness.html
def initialize_seed(seed: int, rank: int = 0):
    seed += rank
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # nll_loss2d_forward_out_cuda_template does not
    # have a deterministic implementation
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)  # for Python random module
    np.random.seed(seed)  # for NumPy
    torch.manual_seed(seed)  # for both CPU and CUDA


def dataloader_kwargs(seed: int, rank: int = 0) -> DataLoaderRandomKw:
    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {
        "worker_init_fn": seed_worker,
        "generator": generator,
    }


def get_module_version(module_list: list[str] = []) -> dict[str, str]:
    version = {}
    for module_name in module_list or MODULES:
        module = importlib.import_module(module_name)
        version[module_name] = module.__version__
    return version
