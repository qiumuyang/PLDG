import random
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset

from .domain import DomainDataset


class FairDGDataset(Dataset):
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
        split: Literal["train", "val", "traintarget"],
        class_to_domain: dict[int, list[int]],
        transform=None,
        use_shm_cache: bool = True,
    ):
        class_to_domain.pop(0, None)  # remove background class
        self.cls = cls
        self.root = root
        self.num_domains = num_domains
        self.target_domain = target_domain
        self.split = split
        self.transform = transform
        self.use_shm_cache = use_shm_cache
        self.class_to_domain = class_to_domain

        match split:
            case "val":
                self.datasets = [self.cls(root, target_domain, "val")]
            case "train":
                self.datasets = [
                    self.cls(root, domain, "train")
                    for domain in range(num_domains) if domain != target_domain
                ]
            case "traintarget":
                self.datasets = [self.cls(root, target_domain, "train")]
            case _:
                raise ValueError(f"invalid mode: {split}")

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @property
    def samples_per_domain(self) -> list[int]:
        return [len(dataset) for dataset in self.datasets]

    def update_voxels(self, class_id: int, num_voxels: int):
        return

    def __getitem__(self, sp: int) -> dict:
        # ensure classwise fairness
        class_id = sp % len(self.class_to_domain) + 1  # skip background
        # weights = self._get_classwise_weight()
        # class_id = random.choices(range(len(self.class_to_domain)),
        #                           weights)[0] + 1

        domain_id = random.choice(self.class_to_domain[class_id])
        dataset = self.datasets[domain_id]
        index = random.randint(0, len(dataset) - 1)

        if self.use_shm_cache:
            from .sharearray import cache
            cache_key = f"{self.cls.__name__}_{dataset.domain}_{index}"
            img = cache(id=cache_key + "_im",
                        array_or_callback=lambda: dataset[index][0],
                        verbose=False)
            mask = cache(id=cache_key + "_lb",
                         array_or_callback=lambda: dataset[index][1],
                         verbose=False)
            img_path = dataset.get_path(index)
        else:
            img, mask, img_path = dataset[index]

        sample = {
            "index": index,
            "image": img,
            "label": mask,
            "domain_id": dataset.domain,
            "rel_domain_id": domain_id,
            "path": img_path,
            "class_id": class_id,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __repr__(self) -> str:
        header = f"{self.cls.__name__}/{self.split} ({len(self)})\n"
        domains = "\n".join(["- " + str(dataset) for dataset in self.datasets])
        return header + domains
