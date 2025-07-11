from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, SubsetRandomSampler

from .dg_dataset import DGDataset


def shuffled(items, shuffled) -> Iterator:
    indices = np.arange(len(items))
    if shuffled:
        np.random.shuffle(indices)
    for i in indices:
        yield items[i]


class MultiDomainSampler(Sampler):
    """Multi-domain sampler for DG dataset.

    The sampler will create a subset sampler for each domain, and sample from
    them in a round-robin(balanced) or random(unbalanced) manner.

    Args:
        lengths: The number of samples in each domain.
        shuffle: Whether to shuffle the order of domains.
        balanced: Whether to sample in a round-robin manner.
    """

    def __init__(
        self,
        lengths: list[int],
        shuffle: bool = True,
        balanced: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.lengths = lengths
        self.base_index = np.cumsum(lengths)
        self.shuffle = shuffle
        self.balanced = balanced
        self.generator = generator
        if balanced:
            self.sub_samplers = [
                iter(SubsetRandomSampler(range(length), generator=generator))
                for length in lengths
            ]
        else:
            self.total_length = sum(lengths)
            self.global_sampler = SubsetRandomSampler(range(self.total_length),
                                                      generator=generator)

    def __iter__(self) -> Iterator:
        if self.balanced:
            stopped = [False] * len(self.sub_samplers)
            while True:
                for i, sampler in shuffled(list(enumerate(self.sub_samplers)),
                                           self.shuffle):
                    base = self.base_index[i - 1] if i > 0 else 0
                    try:
                        yield next(sampler) + base
                    except StopIteration:
                        stopped[i] = True
                        if all(stopped):
                            return
                        self.sub_samplers[i] = iter(
                            SubsetRandomSampler(range(self.lengths[i]),
                                                generator=self.generator))
                        yield next(self.sub_samplers[i]) + base
        else:
            yield from self.global_sampler

    @classmethod
    def from_dataset(
        cls,
        dataset: DGDataset,
        shuffle: bool = True,
        balanced: bool = True,
        generator: torch.Generator | None = None,
    ):
        return cls(dataset.samples_per_domain, shuffle, balanced, generator)


class DistributedMultiDomainSampler(Sampler):

    def __init__(
        self,
        lengths: list[int],
        shuffle: bool = True,
        balanced: bool = True,
        seed: int = 0,
    ):
        self.lengths = lengths
        self.base_index = np.cumsum(lengths)
        self.shuffle = shuffle
        self.balanced = balanced

        self.epoch = 0
        self.seed = seed

        # Total length for unbalanced sampling
        self.total_length = sum(lengths)

        if balanced:
            # Create sub-samplers for each domain
            self.sub_samplers = [
                SubsetRandomSampler(range(length)) for length in lengths
            ]
        else:
            # Global sampler for unbalanced mode
            self.global_sampler = SubsetRandomSampler(range(self.total_length))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.balanced:
            for sampler in self.sub_samplers:
                sampler.generator = g
            iters = [iter(sampler) for sampler in self.sub_samplers]
            stop = [False] * len(iters)
            while True:
                indices = torch.randperm(len(iters), generator=g).tolist()
                for i in indices:
                    try:
                        yield next(iters[i]) + (self.base_index[i - 1]
                                                if i > 0 else 0)
                    except StopIteration:
                        stop[i] = True
                        if all(stop):
                            return
                        iters[i] = iter(self.sub_samplers[i])
                        yield next(iters[i]) + (self.base_index[i - 1]
                                                if i > 0 else 0)
        else:
            self.global_sampler.generator = g
            yield from self.global_sampler

    @classmethod
    def from_dataset(
        cls,
        dataset: DGDataset,
        shuffle: bool = True,
        balanced: bool = True,
        seed: int = 0,
    ):
        return cls(dataset.samples_per_domain, shuffle, balanced, seed)


def cycle(iterable, *manual_set_epoch_keys: str):
    """Creates an infinite loop over the given iterable, setting the epoch for
    the iterable (if `set_epoch` method exists) and additional attributes of
    the iterable specified by `manual_set_epoch_keys`.

    Args:
        iterable: The iterable to cycle through.
        *manual_set_epoch_keys: Additional attribute names of the iterable
                                that also require epoch setting.

    Yields:
        The next item from the iterable in an infinite loop.

    Notes:
        This function is useful for ensuring that the epoch is set correctly
        on samplers and batch samplers, especially when using accelerators
        that may not call set_epoch on the sampler directly.
    """

    epoch = 0  # one epoch is one full pass through the iterable
    while True:
        if hasattr(iterable, "set_epoch"):
            iterable.set_epoch(epoch)
            for key in manual_set_epoch_keys:
                if hasattr(iterable, key):
                    getattr(iterable, key).set_epoch(epoch)
        for item in iterable:
            yield item
        epoch += 1
