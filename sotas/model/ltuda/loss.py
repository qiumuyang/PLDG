import torch
import torch.nn as nn
import torch.nn.functional as F


class pBCE(nn.Module):

    def __init__(self, num_classes=4, **kwargs):
        super(pBCE, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        for i in range(self.num_classes):
            ce_loss = self.criterion(predict[:, i], target[:, i])
            mask = target[:, i] != -1  #ignore unknown
            ce_loss = ce_loss * mask
            ce_loss = torch.mean(ce_loss, dim=[1, 2, 3])
            ce_loss_avg = ce_loss[torch.mean(target[:, i]) != -1].sum(
            ) / ce_loss[torch.mean(target[:, i]) != -1].shape[0]
            total_loss.append(ce_loss_avg)
        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


class pDice(nn.Module):

    def __init__(self, num_classes=4, **kwargs):
        super(pDice, self).__init__()
        self.num_classes = num_classes
        self.eps = 1e-5

    def _dice_loss(self, score, target, mask):
        target = target.float()
        intersect = (score[mask] * target[mask]).sum()
        y_sum = (target[mask] * target[mask]).sum()
        z_sum = (score[mask] * score[mask]).sum()
        dc = (2 * intersect + self.eps) / (z_sum + y_sum + self.eps)
        return 1 - dc

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        p = F.sigmoid(predict)
        for i in range(self.num_classes):
            dice_loss = self._dice_loss(p[:, i], target[:, i], target[:, i]
                                        != -1)
            total_loss.append(dice_loss)
        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]
        return total_loss.sum() / self.num_classes
