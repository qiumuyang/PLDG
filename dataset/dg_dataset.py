from copy import deepcopy
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset

from .domain import DomainDataset


class DGDataset(Dataset):
    """Dataset for Domain Generalization with augmentation.

    Attributes:
        name: dataset name
        target_domain: id of target domain
        mode: train/val/traintarget
            if train, load all domains except target domain
            if val, load target domain only
            if traintarget, load only target domain
        resize: resize image when training
    """

    def __init__(
        self,
        cls: type[DomainDataset],
        root: str | Path,
        num_domains: int,
        target_domain: int,
        split: Literal["train", "val", "traintarget", "val-single"],
        transform=None,
        use_shm_cache: bool = True,
    ):
        self.cls = cls
        self.root = root
        self.num_domains = num_domains
        self.target_domain = target_domain
        self.split = split
        self.transform = transform
        self.use_shm_cache = use_shm_cache

        match split:
            case "val":
                self.datasets = [self.cls(root, target_domain, "val")]
            case "train":
                self.datasets = [
                    self.cls(root, domain, "train")
                    for domain in range(num_domains) if domain != target_domain
                ]
            case "traintarget":
                self.datasets = [self.cls(root, target_domain, "train-single")]
            case "val-single":
                self.datasets = [self.cls(root, target_domain, "val-single")]
            case _:
                raise ValueError(f"invalid mode: {split}")

    def with_transform(self, transform):
        return DGDataset(
            self.cls,
            self.root,
            self.num_domains,
            self.target_domain,
            self.split,  # type: ignore
            transform,
            self.use_shm_cache)

    def copy(self):
        return DGDataset(
            self.cls,
            self.root,
            self.num_domains,
            self.target_domain,
            self.split,  # type: ignore
            deepcopy(self.transform),
            self.use_shm_cache)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @property
    def samples_per_domain(self) -> list[int]:
        return [len(dataset) for dataset in self.datasets]

    def __getitem__(self, index: int) -> dict:
        cnt = 0
        domain_id = -1
        for domain_id, dataset in enumerate(self.datasets):
            if index < cnt + len(dataset):
                break
            cnt += len(dataset)
        else:
            raise IndexError(f"out of range: {index}")

        if self.use_shm_cache:
            from .sharearray import cache
            cache_key = f"{self.cls.__name__}_{dataset.domain}_{index - cnt}"
            img = cache(id=cache_key + "_im",
                        array_or_callback=lambda: dataset[index - cnt][0],
                        verbose=False)
            mask = cache(id=cache_key + "_lb",
                         array_or_callback=lambda: dataset[index - cnt][1],
                         verbose=False)
            img_path = dataset.get_path(index - cnt)
        else:
            img, mask, img_path = dataset[index - cnt]

        sample = {
            "index": index,
            "image": img,
            "label": mask,
            "domain_id": dataset.domain,
            "rel_domain_id": domain_id,
            "path": img_path,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __repr__(self) -> str:
        header = f"{self.cls.__name__}/{self.split} ({len(self)})\n"
        domains = "\n".join(["- " + str(dataset) for dataset in self.datasets])
        return header + domains


class RepeatDataset(Dataset):
    """
    Repeat dataset for multiple times in case of batch size > len(dataset).

    """

    def __init__(self, dataset, repeat: int):
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]
