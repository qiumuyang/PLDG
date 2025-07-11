import random

import numpy as np
import torch
import torch.nn.functional as F

# try:
#     from skimage.exposure import match_histograms
# except ImportError:
#     match_histograms = None

try:
    from batchgenerators.transforms.abstract_transforms import \
        AbstractTransform
except ImportError:
    AbstractTransform = None

from .dg_dataset import DGDataset


class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict):
        for t in self.transforms:
            if AbstractTransform and isinstance(t, AbstractTransform):
                # batchgenerator transform
                sample = t(**sample)
            else:
                sample = t(sample)
        return sample


class MakeCopy:

    def __init__(self, keys: list[str] | dict[str, list[str]] | None = None):
        keys = keys or {}
        if isinstance(keys, list):
            keys = {k: [k] for k in keys}
        self.keys = keys

    def __call__(self, sample: dict):
        for key, targets in self.keys.items():
            for target in targets:
                sample[target] = sample[key].copy()
        return sample


class PadTo:

    def __init__(self,
                 *size: int,
                 add_pad_info: bool = False,
                 skip_keys: list[str] = []):
        self.size = size
        self.add_pad_info = add_pad_info
        self.skip_keys = skip_keys

    def __call__(self, sample: dict):
        pad_info = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and key not in self.skip_keys:
                sample[key] = self.pad_to_size(value, self.size)
                if self.add_pad_info:
                    pad = self.calculate_padding(value, self.size)
                    # expand to 1d tuple
                    pad_info[key + "_pad"] = torch.tensor(sum(pad, ()))
        sample.update(pad_info)
        return sample

    @classmethod
    def calculate_padding(cls, a: np.ndarray,
                          size: tuple[int, ...]) -> list[tuple[int, int]]:
        """Calculate padding to pad array to size."""
        padding = [(max(0, (s1 - s2) // 2),
                    max(0, s1 - s2 - (s1 - s2) // 2)) if s1 > s2 else (0, 0)
                   for s1, s2 in zip(size, a.shape[-len(size):])]
        return padding

    @classmethod
    def pad_to_size(cls, a: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
        """Pad array to size with mode."""
        assert len(size) <= len(
            a.shape), (f"Expected at least {len(size)}D array, "
                       f"got {len(a.shape)}D array.")
        padding = cls.calculate_padding(a, size)
        if any(p[0] > 0 or p[1] > 0 for p in padding):
            padding = [(0, 0)] * (len(a.shape) - len(size)) + padding
            a = np.pad(a, padding, mode="constant", constant_values=0)
        return a

    @classmethod
    def restore_padding(
        cls,
        batch: torch.Tensor,
        pad: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Remove padding from tensor."""
        restored = []
        for t, p in zip(batch, pad):
            dim = len(t.shape)
            dim_p = len(p) // 2
            for i in range(dim_p):
                pad_start, pad_end = p[i * 2], p[i * 2 + 1]
                actual_length: int = t.shape[
                    i + dim - dim_p] - pad_start - pad_end  # type: ignore
                t = torch.narrow(t,
                                 i + dim - dim_p,
                                 start=pad_start,
                                 length=actual_length)
            restored.append(t[None])  # restore batch dim
        return restored


class RandomCrop:

    def __init__(self, *size: int, skip_keys: list[str] = []):
        self.size = size
        self.skip_keys = skip_keys

    def __call__(self, sample: dict):
        rand = None  # to keep same crop for all arrays in one sample
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and key not in self.skip_keys:
                sample[key], rand = self._crop(value, starts=rand)
        return sample

    def _crop(self, arr: np.ndarray, starts=None):
        """Crop expected size from arr.

        arr: np.ndarray[d1, d2, ..., dn]
        """
        assert len(self.size) <= len(
            arr.shape), (f"Expected at least {len(self.size)}D array, "
                         f"got {len(arr.shape)}D array.")
        arr = PadTo.pad_to_size(arr, self.size)

        if starts is None:
            starts = [
                random.randint(0, s1 - s2)  # 0 to (arr size - crop size)
                for s1, s2 in zip(arr.shape[-len(self.size):], self.size)
            ]

        keep = [slice(None)] * (len(arr.shape) - len(self.size))
        crop = [slice(s, s + c) for s, c in zip(starts, self.size)]
        return arr[tuple(keep + crop)], starts


class ForegroundCrop(RandomCrop):

    def __init__(self,
                 *size: int,
                 key: str = "label",
                 p: float = 0.9,
                 skip_keys: list[str] = []):
        self.size = size
        self.key = key
        self.p = p
        self.skip_keys = skip_keys

    def __call__(self, sample: dict):
        if random.random() >= self.p:
            return RandomCrop.__call__(self, sample)
        label = PadTo.pad_to_size(sample[self.key], self.size)
        class_id = sample.get("class_id", -1)
        bbox = self._find_foreground_bbox(label, target_class=class_id)
        if bbox is None:
            return RandomCrop.__call__(self, sample)
        starts = self._sample_crop(label.shape, bbox)  # type: ignore
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and key not in self.skip_keys:
                sample[key], _ = self._crop(value, starts=starts)
        return sample

    def _sample_crop(self, shape, bbox: tuple[tuple[int, int], ...]):
        # sample point inside bbox
        starts = []
        margin = 20  # in case always sample the same point
        for (s1, s2), shape_, size in zip(bbox, shape, self.size):
            start, end = s1, s2 - size
            start = max(0, min(start, shape_ - size))
            # end = max(start, min(end, shape_ - size))
            end = min(shape_ - size, max(end, start + margin))
            starts.append(random.randint(start, end))
        return starts

    def _find_foreground_bbox(self,
                              label: np.ndarray,
                              *,
                              target_class: int = -1):
        """Find bounding box of foreground in label."""

        if target_class > 0 and target_class in np.unique(label):
            foreground = label == target_class
        else:
            foreground = label > 0
        if not np.any(foreground):
            return None

        coords = np.where(foreground)
        d_min, d_max = np.min(coords[0]), np.max(coords[0])
        h_min, h_max = np.min(coords[1]), np.max(coords[1])
        w_min, w_max = np.min(coords[2]), np.max(coords[2])

        return ((d_min, d_max), (h_min, h_max), (w_min, w_max))


class RandomFlip:

    def __init__(self, prob: float, axis: int, *, keys: tuple[str, ...] = ()):
        self.axis = axis
        self.prob = prob
        self.keys = keys

    def __call__(self, sample: dict):
        if random.random() < self.prob:
            for key, value in sample.items():
                if self.keys and key not in self.keys:
                    continue
                if isinstance(value, np.ndarray):
                    # this causes negative strides
                    sample[key] = np.flip(value, axis=self.axis)
        return sample


class DropKeys:

    def __init__(self, *keys):
        self.keys = keys

    def __call__(self, sample: dict):
        for key in self.keys:
            sample.pop(key, None)
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict):
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value.copy()).float()
        return sample


# class MatchHistogram:
#     """Use in-batch domain information to match histogram."""

#     def __init__(self, ref: DGDataset, domain_key: str = "rel_domain_id"):
#         self.ref = ref
#         self.domain_key = domain_key

#     def __call__(self, sample: dict):
#         # sample the reference domain
#         domain_id = sample[self.domain_key]
#         ref_domain_id = random.choice(
#             [i for i in range(self.ref.num_domains - 1) if i != domain_id])
#         # sample the reference image
#         start = sum(self.ref.samples_per_domain[:ref_domain_id])
#         stop = start + self.ref.samples_per_domain[ref_domain_id]
#         ref_sample_id = random.randint(start, stop - 1)
#         ref_sample = self.ref[ref_sample_id]
#         # perform match
#         aug_image = match_histograms(sample["image"], ref_sample["image"])
#         sample["image_hist"] = aug_image
#         sample["ref_domain"] = ref_domain_id
#         return sample


class FilterLabel:
    """Keep only specified classes in label.

    Args:
        class_names: list of class names
        class_of_interest: list of class names to keep
        compress: whether to compress the label to 0, 1, 2, ...
    """

    def __init__(
        self,
        class_names: list[str],
        class_of_interest: list[list[str]],
        domain_key: str,
        *,
        masked_value: int = 0,
        compress: bool = False,
        keep_original: bool = False,
        add_class_ids: bool = False,
    ):
        self.class_names = class_names
        self.class_of_interest = class_of_interest
        self.domain_key = domain_key
        self.masked_value = masked_value
        self.compress = compress
        self.keep_original = keep_original
        self.add_class_ids = add_class_ids

        self.label_maps = [
            self.make_label_map(
                class_names,
                c,
                masked_value=masked_value,
                compress=compress,
            ) for c in class_of_interest
        ]

    @classmethod
    def make_label_map(cls,
                       class_names: list[str],
                       class_of_interest: list[str],
                       masked_value: int = 0,
                       compress: bool = False) -> np.ndarray:
        # integrity check
        for name in class_of_interest:
            assert name in class_names, f"{name} not in class_names"
        label_map = []
        filtered = 0
        for i, name in enumerate(class_names):
            if name in class_of_interest:
                label_map.append(i if not compress else filtered)
                filtered += 1
            else:
                label_map.append(masked_value)
        return np.array(label_map)

    def __call__(self, sample: dict):
        domain = sample[self.domain_key]
        label_map = self.label_maps[domain]
        filtered = label_map[sample["label"]]
        if self.keep_original:
            sample["label_original"] = sample["label"]
        sample["label"] = filtered
        if self.add_class_ids:
            sample["class_ids"] = np.unique(label_map).tolist()
            sample["classes"] = label_map
        return sample


class AddCutmixBox:

    def __init__(
        self,
        p: float = 0.5,
        *,
        vol_min: float = 0.02,
        vol_max: float = 0.3,
        r: float = 0.3,
        key: str = "cutmix_box",
    ):
        self.p = p
        self.vol_min = vol_min
        self.vol_max = vol_max
        self.r = r
        self.key = key

    def __call__(self, sample: dict):
        d, h, w = sample["image"].shape[-3:]
        mask = torch.zeros((d, h, w), dtype=torch.long)
        if random.random() < self.p:
            vol = random.uniform(self.vol_min, self.vol_max) * d * h * w
            while True:
                x = random.randint(0, w)
                y = random.randint(0, h)
                z = random.randint(0, d)
                r1 = random.uniform(self.r, 1 / self.r)
                r2 = random.uniform(self.r, 1 / self.r)
                box_d = int(np.cbrt(vol / (r1 * r2)))
                box_h = int(r1 * box_d)
                box_w = int(r2 * box_d)
                # check if box is inside image
                if (x + box_w < w and y + box_h < h and z + box_d < d):
                    break
            mask[z:z + box_d, y:y + box_h, x:x + box_w] = 1
        sample[self.key] = mask
        return sample


class PatchCutout:

    def __init__(self, patch: int, ratio: float = 0.3):
        self.patch = patch
        self.ratio = ratio

    def __call__(self, sample: dict):
        d, h, w = sample["image"].shape[-3:]
        pd, ph, pw = d // self.patch, h // self.patch, w // self.patch
        num_patch = pd * ph * pw
        num_remove = int(num_patch * self.ratio)
        remove_id = random.sample(range(num_patch), num_remove)
        mask = torch.zeros((d, h, w), dtype=torch.long)
        for i in remove_id:
            z, y, x = i // (ph * pw), (i // pw) % ph, i % pw
            mask[z * pd:(z + 1) * pd, y * ph:(y + 1) * ph,
                 x * pw:(x + 1) * pw] = 1
        sample["cutout_mask"] = mask
        return sample
