# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F

affine_par = True

import torch.distributed as dist
from einops import rearrange, repeat
from timm.layers.weight_init import trunc_normal_

from model.vnet import VNet


def momentum_update(old_value, new_value, momentum, debug=False):
    momentum = torch.tensor(momentum).cuda()
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|"
            .format(momentum, torch.norm(old_value, p=2), (1 - momentum),
                    torch.norm(new_value, p=2), torch.norm(update, p=2)))
    return update


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class ProjectionHead(nn.Module):

    def __init__(self,
                 dim_in,
                 proj_dim=256,
                 proj='convmlp',
                 bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv3d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm3d(dim_in), nn.ReLU(inplace=True),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class VNetProto(nn.Module):

    def __init__(self, n_channels, n_classes, num_prototype=1):
        super(VNetProto, self).__init__()
        self.vnet = VNet(n_channels, n_classes, 32)
        self.num_classes = n_classes + 1

        # prototype
        self.gamma = 0.999,
        self.num_prototype = num_prototype
        self.update_prototype = True
        self.pretrain_prototype = False

        in_channels = self.vnet.feat_channels[-1]
        self.prototypes_labeled = nn.Parameter(torch.zeros(
            self.num_classes, self.num_prototype, in_channels),
                                               requires_grad=True)
        self.prototypes_unlabeled = nn.Parameter(torch.zeros(
            self.num_classes, self.num_prototype, in_channels),
                                                 requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes_labeled, std=0.02)
        trunc_normal_(self.prototypes_unlabeled, std=0.02)

    def prototype_learning(self,
                           _c,
                           out_seg,
                           gt_seg,
                           masks,
                           labeled=True):  #update prototype
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        if labeled:
            cosine_similarity = torch.mm(
                _c,
                self.prototypes_labeled.view(
                    -1, self.prototypes_labeled.shape[-1]).t())
        else:
            cosine_similarity = torch.mm(
                _c,
                self.prototypes_unlabeled.view(
                    -1, self.prototypes_unlabeled.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        if labeled:
            protos = self.prototypes_labeled.data.clone()
        else:
            protos = self.prototypes_unlabeled.data.clone()

        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]

            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:

                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :],
                                            new_value=f[n != 0, :],
                                            momentum=self.gamma,
                                            debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype *
                                                          k)

        if labeled:
            self.prototypes_labeled = nn.Parameter(l2_normalize(protos),
                                                   requires_grad=False)
        else:
            self.prototypes_unlabeled = nn.Parameter(l2_normalize(protos),
                                                     requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            if labeled:
                protos = self.prototypes_labeled.data.clone()
            else:
                protos = self.prototypes_unlabeled.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            if labeled:
                self.prototypes_labeled = nn.Parameter(protos,
                                                       requires_grad=False)
            else:
                self.prototypes_unlabeled = nn.Parameter(protos,
                                                         requires_grad=False)

        return proto_logits, proto_target

    def forward(self,
                x,
                gt_seg=None,
                pseudo_seg=None,
                pretrain_prototype=False,
                use_prototype=False,
                test_linear=False,
                test_lproto=False,
                test_ulproto=False):
        logits, feat = self.vnet(x, return_feats=True)
        x_out = feat[-1]

        if test_linear:
            c = self.proj_head(x_out)
            return logits, c

        if test_lproto or test_ulproto:
            feats = x_out

            c = self.proj_head(feats)  # use prototype for seg
            _c = rearrange(c, 'b c d h w -> (b d h w) c')
            _c = self.feat_norm(_c)
            _c = l2_normalize(_c)

            # test unlabeled proto
            if test_ulproto:
                self.prototypes_unlabeled.data.copy_(
                    l2_normalize(self.prototypes_unlabeled))

                # n: h*w, k: num_class, m: num_prototype
                masks = torch.einsum('nd,kmd->nmk', _c,
                                     self.prototypes_unlabeled)
            elif test_lproto:
                # # test labeled proto
                self.prototypes_labeled.data.copy_(
                    l2_normalize(self.prototypes_labeled))

                # n: h*w, k: num_class, m: num_prototype
                masks = torch.einsum('nd,kmd->nmk', _c,
                                     self.prototypes_labeled)

            out_seg = torch.amax(masks, dim=1)
            out_seg = self.mask_norm(out_seg)
            out_seg = rearrange(out_seg,
                                "(b d h w) k -> b k d h w",
                                b=feats.shape[0],
                                d=feats.shape[2],
                                h=feats.shape[3])
            return out_seg, c

        if not use_prototype:
            return logits

        feats = x_out

        c = self.proj_head(feats)  # use prototype for seg
        _c = rearrange(c, 'b c d h w -> (b d h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes_labeled.data.copy_(
            l2_normalize(self.prototypes_labeled))
        self.prototypes_unlabeled.data.copy_(
            l2_normalize(self.prototypes_unlabeled))

        # n: h*w, k: num_class, m: num_prototype
        masks_labeled = torch.einsum('nd,kmd->nmk', _c,
                                     self.prototypes_labeled)
        out_seg_labeled = torch.amax(masks_labeled, dim=1)
        out_seg_labeled = self.mask_norm(out_seg_labeled)
        out_seg_labeled = rearrange(out_seg_labeled,
                                    "(b d h w) k -> b k d h w",
                                    b=feats.shape[0],
                                    d=feats.shape[2],
                                    h=feats.shape[3])

        masks_unlabeled = torch.einsum('nd,kmd->nmk', _c,
                                       self.prototypes_unlabeled)
        out_seg_unlabeled = torch.amax(masks_unlabeled, dim=1)
        out_seg_unlabeled = self.mask_norm(out_seg_unlabeled)
        out_seg_unlabeled = rearrange(out_seg_unlabeled,
                                      "(b d h w) k -> b k d h w",
                                      b=feats.shape[0],
                                      d=feats.shape[2],
                                      h=feats.shape[3])

        #use pseduo_seg to replace gt_seg
        if pretrain_prototype is False and use_prototype is True and gt_seg is not None:
            gt_seg = F.interpolate(gt_seg.unsqueeze(1).float(),
                                   size=feats.size()[2:],
                                   mode='nearest').view(-1)
            pseudo_seg = F.interpolate(pseudo_seg.unsqueeze(1).float(),
                                       size=feats.size()[2:],
                                       mode='nearest').view(-1)
            contrast_logits_labeled, contrast_target_labeled = self.prototype_learning(
                _c, out_seg_labeled, gt_seg, masks_labeled, labeled=True)
            contrast_logits_unlabeled, contrast_target_unlabeled = self.prototype_learning(
                _c,
                out_seg_unlabeled,
                pseudo_seg,
                masks_unlabeled,
                labeled=False)
            return out_seg_labeled, contrast_logits_labeled, contrast_target_labeled, out_seg_unlabeled, contrast_logits_unlabeled, contrast_target_unlabeled
