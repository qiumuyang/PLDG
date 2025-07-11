#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18, 2023

@author: xychen
"""

import torch
import torch.nn as nn


class BalancedCELoss(nn.Module):

    def __init__(self,
                 n_classes,
                 beta=1.0,
                 gamma=2.0,
                 multiplier_for_unlabeled_data=3.0,
                 epsilon=1e-6):
        super(BalancedCELoss, self).__init__()
        self.n_classes = n_classes
        self.beta = beta
        self.gamma = gamma
        self.multiplier_for_unlabeled_data = multiplier_for_unlabeled_data
        self.epsilon = epsilon

    def forward(self, probs, target, annotated_fg_categories):

        regularization = 0.
        cross_entropy = []

        batchSize = target.size(0)
        for i in range(batchSize):
            probs_i = probs[i]
            target_i = target[i]

            unique_labels = torch.sort(torch.unique(target_i)).values

            if unique_labels.size(0) == 1 and unique_labels[0] == 0:
                regularization -= self.multiplier_for_unlabeled_data * torch.mean(
                    torch.sum(probs_i * torch.log(
                        torch.clamp(probs_i, self.epsilon, 1 - self.epsilon)),
                              dim=0))
            else:
                regularization -= torch.mean(
                    torch.sum(probs_i * torch.log(
                        torch.clamp(probs_i, self.epsilon, 1 - self.epsilon)),
                              dim=0))

            annotated_fg_categories_i = annotated_fg_categories[i]
            annotated_fg_categories_i = annotated_fg_categories_i[
                annotated_fg_categories_i > 0]

            unique_labels_in_this_patch = torch.sort(
                torch.unique(target_i)).values

            probs_i = torch.permute(probs_i, (1, 2, 3, 0))
            probs_i_size = probs_i.size()
            probs_i = probs_i.reshape(
                (probs_i_size[0] * probs_i_size[1] * probs_i_size[2],
                 probs_i_size[3]))
            target_i = torch.flatten(target_i)

            cross_entropy_i = []
            for j in range(unique_labels_in_this_patch.size(0)):
                unique_label = unique_labels_in_this_patch[j]
                indices = torch.where(target_i == unique_label)[0]

                if unique_label == 0:
                    unannotated_categories_or_zero = []
                    for k in range(self.n_classes):
                        if k not in annotated_fg_categories_i:
                            unannotated_categories_or_zero.append(k)

                    probs_i_j = probs_i[
                        indices][:, unannotated_categories_or_zero]
                    sum_probs = torch.sum(probs_i_j, dim=1)

                    cross_entropy_i.append(
                        -torch.pow(1 - sum_probs, self.gamma) * torch.log(
                            torch.clamp(sum_probs, self.epsilon,
                                        1 - self.epsilon)))
                else:
                    probs_i_j = probs_i[indices][:, unique_label.int()]

                    cross_entropy_i.append(
                        -torch.pow(1 - probs_i_j, self.gamma) * torch.log(
                            torch.clamp(probs_i_j, self.epsilon,
                                        1 - self.epsilon)))

            cross_entropy.append(
                torch.mean(torch.concat(cross_entropy_i, dim=0)))

        if len(cross_entropy) == 0:
            return torch.tensor(0.).to(
                probs.device), regularization / batchSize
        else:
            return torch.mean(
                torch.stack(cross_entropy)), regularization / batchSize


class DiceLoss(nn.Module):

    def __init__(self, n_classes, smooth=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        loss = 1 - loss
        return loss

    def _dice_loss_v2(self, score, target):
        target = target.float()
        tp = torch.sum(score * target)
        fp = torch.sum(score * (1 - target))
        fn = torch.sum((1 - score) * target)
        # tn = torch.sum((1 - score) * (1 - target))
        loss = 1 - torch.mean(
            (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth))
        return loss

    def _dice_loss_no_smooth(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = 2 * intersect / (z_sum + y_sum)
        loss = 1 - loss
        return loss

    def forward(self, probs, target, annotated_fg_categories, weight=None):

        one_hot = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert probs.size() == one_hot.size(
        ), 'predict {} & target {} shape do not match'.format(
            probs.size(), one_hot.size())

        dice_loss = []

        batchSize = target.size(0)
        for i in range(batchSize):
            probs_i = probs[i]
            one_hot_i = one_hot[i]
            target_i = target[i]

            probs_list = []
            one_hot_list = []

            # select the i-th sample
            annotated_fg_categories_i = annotated_fg_categories[i]
            # remove the background label (0)
            annotated_fg_categories_i = annotated_fg_categories_i[
                annotated_fg_categories_i > 0]

            probs_i_size = probs_i.size()
            probs_i = torch.permute(probs_i, (1, 2, 3, 0))
            probs_i = probs_i.reshape(
                (probs_i_size[1] * probs_i_size[2] * probs_i_size[3],
                 probs_i_size[0]))
            one_hot_i = torch.permute(one_hot_i, (1, 2, 3, 0))
            one_hot_i = one_hot_i.reshape(
                (probs_i_size[1] * probs_i_size[2] * probs_i_size[3],
                 probs_i_size[0]))
            target_i = torch.flatten(target_i)

            unique_labels_in_this_patch = torch.sort(
                torch.unique(target_i)).values

            # i.e., Marginal loss (MedIA'21)
            for j in range(unique_labels_in_this_patch.size(0)):
                unique_label = unique_labels_in_this_patch[j]

                if unique_label == 0:
                    # unannotated categories or background
                    unannotated_categories_or_zero = []
                    for k in range(self.n_classes):
                        if k not in annotated_fg_categories_i:
                            unannotated_categories_or_zero.append(k)

                    probs_i_j = probs_i[:, unannotated_categories_or_zero]
                    sum_probs = torch.sum(probs_i_j, dim=1)

                    one_hot_i_j = one_hot_i[:, unannotated_categories_or_zero]
                    sum_one_hot = torch.sum(one_hot_i_j, dim=1)

                    probs_list.append(sum_probs)
                    one_hot_list.append(sum_one_hot)
                else:
                    probs_i_j = probs_i[:, unique_label.int()]
                    one_hot_i_j = one_hot_i[:, unique_label.int()]

                    probs_list.append(probs_i_j)
                    one_hot_list.append(one_hot_i_j)

            probs_stack = torch.stack(probs_list)
            one_hot_stack = torch.stack(one_hot_list)

            tp = torch.sum(probs_stack * one_hot_stack, dim=1)
            fp = torch.sum(probs_stack * (1 - one_hot_stack), dim=1)
            fn = torch.sum((1 - probs_stack) * one_hot_stack, dim=1)
            # tn = torch.sum((1 - probs_stack) * (1 - one_hot_stack), dim=1)

            dice_loss.append(1 - torch.mean((2 * tp + self.smooth) /
                                            (2 * tp + fp + fn + self.smooth)))

        if len(dice_loss) == 0:
            return torch.tensor(0.).to(probs.device)
        else:
            return torch.mean(torch.stack(dice_loss))
