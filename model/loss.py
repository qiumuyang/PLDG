from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, n_classes, p: Literal[1, 2] = 2, eps: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.p = p
        self.eps = eps

    @classmethod
    def one_hot(cls, input_tensor, n_classes):
        return F.one_hot(input_tensor,
                         num_classes=n_classes).permute(0, 4, 1, 2, 3).float()

    def _dice_loss(self, score, target, ignore):
        if ignore is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        else:
            mask = ignore != 1
        target = target.float()
        intersect = (score[mask] * target[mask]).sum()
        if self.p == 2:
            y_sum = (target[mask] * target[mask]).sum()
            z_sum = (score[mask] * score[mask]).sum()
        else:
            y_sum = target[mask].sum()
            z_sum = score[mask].sum()
        dc = (2 * intersect + self.eps) / (z_sum + y_sum + self.eps)
        return 1 - dc

    @classmethod
    def dc(cls, pred, target, n_classes, eps=1e-5):
        assert pred.size() == target.size()
        p = cls.one_hot(pred, n_classes)
        q = cls.one_hot(target, n_classes)
        intersect = (p * q).sum(dim=(2, 3, 4))
        y_sum = q.sum(dim=(2, 3, 4))
        z_sum = p.sum(dim=(2, 3, 4))
        return (2 * intersect + eps) / (y_sum + z_sum + eps)

    @classmethod
    def dc_overlap(cls, pred, target, n_classes, eps=1e-5):
        q = cls.one_hot(target, n_classes)
        assert pred.size() == q.size()
        # make sure pred is all 0 or 1
        assert ((pred == 0) | (pred == 1)).all()
        intersect = (pred * q).sum(dim=(2, 3, 4))
        y_sum = q.sum(dim=(2, 3, 4))
        z_sum = pred.sum(dim=(2, 3, 4))
        return (2 * intersect + eps) / (y_sum + z_sum + eps)

    def forward(
        self,
        inputs,
        target,
        softmax="softmax",
        ignore=None,
        onehot=True,
    ):
        """
        Args:
            inputs: (N, C, D, H, W)
            target: (N, D, H, W)
            softmax: how to process the input
            ignore: (N, D, H, W) or (N, C, D, H, W)
            onehot: whether to convert the target (N, H, W) to onehot
        """
        if softmax == "softmax":
            inputs = inputs.softmax(dim=1)
        elif softmax == "sigmoid":
            inputs = inputs.sigmoid()
        elif softmax not in ("", "none", None):
            raise ValueError("softmax should be softmax or sigmoid or empty")

        if onehot:
            target = self.one_hot(target, self.n_classes)

        if ignore is not None and ignore.ndim == 4:
            ignore = ignore.unsqueeze(1).expand(-1, self.n_classes, -1, -1, -1)

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(
                inputs[:, i],
                target[:, i],
                ignore[:, i] if ignore is not None else None,
            )
            loss += dice
        return loss / self.n_classes


class AggregatedDiceLoss(DiceLoss):

    def forward(self, inputs, target, classes):  # type: ignore
        inputs = inputs.softmax(dim=1)
        target = self.one_hot(target, self.n_classes)

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        # aggregate by classes
        keep_mask = classes > 0  # non-background classes are kept, [B, C]
        # otherwise logits summed together
        loss = 0.0
        for x, y, mask in zip(inputs, target, keep_mask):
            agg_x = torch.cat((x[~mask].sum(dim=0)[None], x[mask]), dim=0)
            agg_y = torch.cat((y[~mask].sum(dim=0)[None], y[mask]), dim=0)
            loss += self._dice_loss(agg_x, agg_y, None)
        return loss / len(inputs)


class ExclusionDiceLoss(DiceLoss):

    def convert_exclusion(self, target, classes):
        # target: [D, H, W]
        # bg <=> cls in classes
        # cls in classes <=> all classes
        onehot = F.one_hot(target, num_classes=self.n_classes)
        onehot[target > 0] = 1 - onehot[target > 0]
        onehot[target == 0] = (classes > 0).long()
        return onehot.permute(3, 0, 1, 2).float()

    def forward(self, inputs, target, classes):  # type: ignore
        inputs = inputs.softmax(dim=1)

        loss = 0.0
        for x, y, c in zip(inputs, target, classes):
            z = self.convert_exclusion(y, c)
            # dice_loss = 1 - dice
            # loss = 1 - dice_loss => loss = dice
            # minimize dice between inputs and exclusion
            loss += 1 - self._dice_loss(x, z, None)
        return loss / len(inputs)  # reduce overlap


class AggregatedCrossEntropyLoss(nn.Module):

    def __init__(self, n_classes):
        super(AggregatedCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, classes):
        target = DiceLoss.one_hot(target, self.n_classes)

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        keep_mask = classes > 0
        loss = 0.0
        for x, y, mask in zip(inputs, target, keep_mask):
            agg_x = torch.cat((x[~mask].logsumexp(dim=0)[None], x[mask]),
                              dim=0)[None]  # add batch dim
            agg_y = torch.cat((y[~mask].sum(dim=0)[None], y[mask]),
                              dim=0)[None]
            loss += F.cross_entropy(agg_x, agg_y.argmax(dim=1))
        return loss / len(inputs)


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, n_classes):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, classes):
        target = DiceLoss.one_hot(target, self.n_classes)
        target = target[:, 1:]

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        keep_mask = classes[:, 1:] > 0

        mask = keep_mask.view(*keep_mask.shape, 1, 1, 1).expand_as(inputs)
        return F.binary_cross_entropy_with_logits(inputs[mask], target[mask])

    def forward_ignore(self, inputs, target, ignore):
        target = DiceLoss.one_hot(target, self.n_classes)
        target = target[:, 1:]

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        mask = (ignore != 1).unsqueeze(1).expand_as(inputs)
        return F.binary_cross_entropy_with_logits(inputs[mask], target[mask])


class BinaryDiceLoss(DiceLoss):

    def forward(self, inputs, target, classes):  # type: ignore
        inputs = inputs.sigmoid()
        target = DiceLoss.one_hot(target, self.n_classes)
        target = target[:, 1:]

        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: "
                             f"{inputs.size()}, {target.size()}")

        keep_mask = classes[:, 1:] > 0
        loss = 0.0
        for x, y, mask in zip(inputs, target, keep_mask):
            loss += self._dice_loss(x[mask], y[mask], None)
        return loss / len(inputs)

    def forward_ignore(self, inputs, target, ignore):  # type: ignore
        inputs = inputs.sigmoid()
        target = DiceLoss.one_hot(target, self.n_classes)
        target = target[:, 1:]
        loss = 0.0
        for i in range(self.n_classes - 1):
            loss += self._dice_loss(inputs[:, i], target[:, i], ignore)
        return loss / (self.n_classes - 1)
