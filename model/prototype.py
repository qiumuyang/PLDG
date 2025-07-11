from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast
from torch import Tensor

DOMAIN_MODE = Literal["in-domain", "across-domain", "weighted-across-domain"]
METRIC = Literal["cosine", "l1", "l2"]


class PrototypeBank(nn.Module):
    """A bank of prototypes for multi-domain and multi-class.

    Attributes:
        num_domains (int): Number of domains.
        num_classes (int): Number of classes.
        dim (int): Feature channels of the prototypes.
        momentum (float): Momentum for updating prototypes.
        mean_updates (int): Number of updates to use mean instead of ema.
        accelerator (Accelerator | None): Accelerator for distributed training.
        prototypes (Tensor): Tensor containing the prototype vectors.
        update_counts (Tensor): Tensor tracking the number of updates for each prototype.
    """

    prototypes: Tensor
    update_counts: Tensor

    def __init__(self,
                 num_domains: int,
                 num_classes: int,
                 dim: int,
                 momentum: float = 0.999,
                 mean_updates: int = 1000,
                 *,
                 accelerator: Accelerator | None = None):
        super().__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.mean_updates = mean_updates
        self.accelerator = accelerator

        # yapf: disable
        self.register_buffer("prototypes", torch.zeros(num_domains, num_classes, dim))
        self.register_buffer("update_counts", torch.zeros(num_domains, num_classes, dtype=torch.long))
        # yapf: enable

    def propagate(self,
                  class_to_domain: dict[int, list[int]],
                  skip_background=True):
        """
        Propagate prototypes from class-labeled domains to all other domains.
        """
        for cls, domains in class_to_domain.items():
            if cls == 0 and skip_background:
                continue
            proto = self.prototypes[domains, cls].mean(dim=0)
            count = self.update_counts[domains, cls].float().mean().long()
            # overwrite domains not in source_domains
            for domain in range(self.num_domains):
                if domain not in domains:
                    self.prototypes[domain, cls] = proto
                    self.update_counts[domain, cls] = count

    def get_domain_prototype(
        self,
        batch_size: int,
        mode: DOMAIN_MODE,
        domains: Tensor | None,
    ) -> Tensor:
        """
        Args:
            mode: Prototype from the same domain or all domains.
            domains: Domain IDs for each sample.

        Returns:
            Tensor[batch_size, num_classes, dim]
        """
        if mode == "in-domain":
            if domains is None:
                raise ValueError(
                    "domains must not be None when mode is in-domain")
            p = self.prototypes[domains.long()]
        elif mode == "across-domain":
            p = self.prototypes.mean(dim=0, keepdim=True)
        elif mode == "weighted-across-domain":
            count = self.update_counts.float()
            denom = count.sum(dim=0, keepdim=True) + 1e-6
            weights = (count / denom).unsqueeze(-1)  # [domain, c, 1]
            p = (self.prototypes * weights).sum(dim=0, keepdim=True)
        else:
            raise NotImplementedError
        if p.shape[0] != batch_size:
            p = p.expand(batch_size, -1, -1)
        return p

    def distance(self,
                 features: Tensor,
                 *,
                 domains: Tensor | None = None,
                 mode: DOMAIN_MODE = "in-domain",
                 metric: METRIC = "l2") -> Tensor:
        """
        Compute distance between features and prototypes.

        Args:
            features: Input features with shape (b, dim, d, h, w).
            domains: Domain IDs for each sample.
            mode: Prototype from the same domain or all domains.
            metric: Distance metric to use.

        Returns:
            Tensor[b, c, d, h, w]
            Distance between features and each class prototype.
        """
        b = features.shape[0]
        p = self.get_domain_prototype(b, mode, domains)  # b, c, d
        f = features.permute(0, 2, 3, 4, 1).reshape(b, -1, self.dim)  # b, v, d
        if metric == "cosine":
            f = F.normalize(f, dim=2, p=2)
            p = F.normalize(p, dim=2, p=2)
            d = 1 - torch.bmm(f, p.transpose(1, 2))
        elif metric == "l1":
            d = torch.cdist(f, p, p=1)
        elif metric == "l2":
            d = torch.cdist(f, p, p=2)
        else:
            raise NotImplementedError
        return d.transpose(1, 2).reshape(b, -1, *features.shape[2:])

    def classify(self,
                 features: Tensor,
                 target_size: torch.Size | None = None,
                 *,
                 domains: Tensor | None = None,
                 mode: DOMAIN_MODE = "in-domain",
                 metric: METRIC = "l2",
                 tau: float = 1.0) -> Tensor:
        """
        Compute probability distribution over classes for each input feature.

        Args:
            features: Input features with shape (b, dim, d, h, w).
            target_size: Expect output size of the tensor.
            domains: Domain IDs for each sample.
            mode: Prototype from the same domain or all domains.
            metric: Distance metric to use.
            tau: Temperature scaling parameter.

        Returns:
            Tensor[b, c, d, h, w]
            Probability distribution over classes.
        """
        d = self.distance(features, domains=domains, mode=mode, metric=metric)
        if target_size is not None:
            d = F.interpolate(d, size=target_size, mode="trilinear")
        return (-d / tau).softmax(dim=1)

    def weight_label(self,
                     features: Tensor,
                     labels: Tensor,
                     *,
                     domains: Tensor | None = None,
                     mode: DOMAIN_MODE = "in-domain",
                     metric: METRIC = "l2",
                     tau: float = 1.0) -> Tensor:
        """
        Compute the probability-weighted label tensor for given features.

        Args:
            features (Tensor): Input features with shape (b, dim, d, h, w).
            labels (Tensor): Ground truth labels with shape (b, d, h, w).
            domains (Tensor | None, optional): Domain IDs for each sample with shape (b,).
            mode (DOMAIN_MODE, optional): Prototype retrieval mode. Defaults to "in-domain".
            metric (METRIC, optional): Distance metric for classification. Defaults to "l2".
            tau (float, optional): Temperature scaling parameter. Defaults to 1.0.

        Returns:
            Tensor: Probability-weighted label tensor with shape (b, d, h, w).
        """

        prob = self.classify(features,
                             target_size=labels.shape[1:],
                             domains=domains,
                             mode=mode,
                             metric=metric,
                             tau=tau)
        return torch.gather(prob, dim=1,
                            index=labels.long().unsqueeze(1)).squeeze(1)

    def update(self,
               features: Tensor,
               labels: Tensor,
               domains: Tensor,
               *,
               ignores: Tensor | None = None,
               min_feat_voxels: int = 10):
        """
        Update the prototypes based on the given features and labels.

        Args:
            features (Tensor): Input features with shape (b, dim, d, h, w).
            labels (Tensor): Ground truth labels with shape (b, d, h, w).
            domains (Tensor): Domain IDs for each sample with shape (b,).
            ignores (Tensor | None, optional): Ignored label indices. Defaults to None.
            min_feat_voxels (int, optional): Minimum number of voxels in a feature map. Defaults to 10.
        """
        d, h, w = features.shape[-3:]
        labels = F.interpolate(labels.float().unsqueeze(1),
                               size=(d, h, w),
                               mode="nearest").squeeze(1).long()
        if ignores is None:
            ignores = torch.zeros_like(labels)
        else:
            ignores = F.interpolate(ignores.float().unsqueeze(1),
                                    size=(d, h, w),
                                    mode="nearest").squeeze(1).long()

        if self.accelerator is not None:
            features = self.accelerator.gather(features)  # type: ignore
            labels = self.accelerator.gather(labels)  # type: ignore
            domains = self.accelerator.gather(domains)  # type: ignore
            ignores = self.accelerator.gather(ignores)  # type: ignore

        assert ignores is not None

        if self.accelerator is None or self.accelerator.is_local_main_process:
            for domain, feature, label, ignore in zip(domains, features,
                                                      labels, ignores):
                # convert to shape (dhw, c) or (dhw,)
                feature = feature.permute(1, 2, 3, 0).reshape(-1, self.dim)
                label = label.reshape(-1)
                ignore = ignore.reshape(-1)
                # filter out ignored voxels
                feature = feature[ignore == 0]
                label = label[ignore == 0]
                for c in torch.unique(label):
                    label_mask = label == c
                    num_feat_voxels = label_mask.sum().item()
                    if num_feat_voxels < min_feat_voxels:
                        continue
                    feat_mean = feature[label_mask].mean(dim=0)
                    proto_cur = self.prototypes[domain, c]
                    update_cnt = self.update_counts[domain, c]
                    if update_cnt <= self.mean_updates:
                        # use simple average for the first few updates
                        proto = proto_cur * update_cnt + feat_mean
                        proto = proto / (update_cnt + 1)
                    else:
                        # use exponential moving average
                        # yapf: disable
                        proto = (proto_cur * self.momentum +
                                feat_mean * (1 - self.momentum))
                        # yapf: enable
                    self.prototypes[domain, c] = proto
                    self.update_counts[domain, c] += 1

        if self.accelerator is not None:
            self.prototypes = broadcast(self.prototypes)  # type: ignore
            self.update_counts = broadcast(self.update_counts)  # type: ignore

    def compute_contrastive_loss(
        self,
        features: Tensor,
        labels: Tensor,
        domains: Tensor,
        *,
        ignores: Tensor | None = None,
        mode: Literal["in-domain", "across-domain",
                      "in-domain-anti"] = "across-domain",
        temperature: float = 0.1,
        skip_background: bool = True,
        top_k: int = -1,
    ):
        """
        Compute contrastive loss using an InfoNCE-like formulation.

        For each feature, the loss encourages a higher similarity with
            all positive prototypes (same-class from other domains)
        compared to
            negative prototypes (different classes from the same domain).
        """
        b, _, d, h, w = features.shape
        labels = F.interpolate(labels.float().unsqueeze(1),
                               size=(d, h, w),
                               mode="nearest").squeeze(1).long()
        if ignores is None:
            ignores = torch.zeros_like(labels)
        else:
            ignores = cast(
                Tensor,
                F.interpolate(ignores.float().unsqueeze(1),
                              size=(d, h, w),
                              mode="nearest").squeeze(1).long())
        feat = features.permute(0, 2, 3, 4, 1).reshape(-1, self.dim)
        label = labels.reshape(-1)
        domain = domains.view(b, 1, 1, 1, 1).expand(b, d, h, w, 1).reshape(-1)
        ignore = ignores.reshape(-1)
        if skip_background:
            ignore = (ignore > 0) | (label == 0)
        feat = feat[ignore == 0]
        label = label[ignore == 0]
        domain = domain[ignore == 0]

        N = feat.shape[0]
        if N == 0:
            return torch.zeros(1, device=features.device)
        D, C = self.num_domains, self.num_classes

        # collect positive prototypes (mask out the current domain)
        if mode == "across-domain":
            proto_pos = self.prototypes[:, label].permute(1, 0,
                                                          2)  # (N, D, dim)
            domain_mask = (torch.arange(D, device=features.device)
                           != domain.unsqueeze(1))  # (N, D)
            proto_pos = proto_pos[domain_mask].view(N, D - 1, self.dim)
        elif mode == "in-domain" or mode == "in-domain-anti":
            proto_pos = self.prototypes[domain, label].unsqueeze(1)
        else:
            raise NotImplementedError

        # collect negative prototypes
        proto_neg = self.prototypes[domain]  # (N, C, dim)
        if skip_background:
            class_mask = (
                (torch.arange(C, device=features.device) != label.unsqueeze(1))
                & (torch.arange(C, device=features.device) != 0))
            cc = C - 2
        else:
            class_mask = (torch.arange(C, device=features.device)
                          != label.unsqueeze(1))  # (N, C)
            cc = C - 1
        proto_neg = proto_neg[class_mask].view(N, cc, self.dim)

        if mode == "in-domain-anti":
            # add same class diff domain as negative
            domain_mask = (torch.arange(D, device=features.device)
                           != domain.unsqueeze(1))  # (N, D)
            proto_neg_d = self.prototypes[:, label].permute(1, 0, 2)
            proto_neg_d = proto_neg_d[domain_mask].view(N, D - 1, self.dim)
            proto_neg = torch.cat([proto_neg, proto_neg_d], dim=1)

        # normalize features and prototypes
        feat = F.normalize(feat, dim=1)  # (N, dim)
        proto_pos = F.normalize(proto_pos, dim=2)  # (N, D-1, dim)
        proto_neg = F.normalize(proto_neg, dim=2)  # (N, C-1, dim)

        # compute similarity
        sim_pos = torch.einsum("nf,ndf->nd", feat, proto_pos)  # (N, D-1)
        sim_neg = torch.einsum("nf,ncf->nc", feat, proto_neg)  # (N, C-1)
        if top_k > 0:
            # top-k hard negative
            sim_neg, _ = torch.topk(sim_neg, top_k, dim=1)

        # scale similarity by temperature and compute exponentials
        pos = (sim_pos / temperature).exp()
        neg = (sim_neg / temperature).exp()

        numerator = pos.sum(dim=1)
        denominator = numerator + neg.sum(dim=1)
        return -torch.log(numerator / denominator).mean()


class ProtoSingle(PrototypeBank):

    def __init__(self,
                 num_domains: int,
                 num_classes: int,
                 dim: int,
                 momentum: float = 0.999,
                 mean_updates: int = 1000,
                 *,
                 accelerator: Accelerator | None = None):
        # override num_domains
        super().__init__(1,
                         num_classes,
                         dim,
                         momentum,
                         mean_updates,
                         accelerator=accelerator)

    def update(self,
               features: Tensor,
               labels: Tensor,
               domains: Tensor,
               *,
               ignores: Tensor | None = None,
               min_feat_voxels: int = 10):
        super().update(features,
                       labels,
                       torch.zeros_like(domains),
                       ignores=ignores,
                       min_feat_voxels=min_feat_voxels)

    def propagate(self,
                  class_to_domain: dict[int, list[int]],
                  skip_background=True):
        # since we only have a single domain
        # we do not need to propagate
        pass

    def distance(self,
                 features: Tensor,
                 *,
                 domains: Tensor | None = None,
                 mode: DOMAIN_MODE = "in-domain",
                 metric: METRIC = "l2") -> Tensor:
        if domains is not None:
            domains = torch.zeros_like(domains)
        return super().distance(features,
                                domains=domains,
                                mode=mode,
                                metric=metric)

    # classify and weight_label are based on distance
    # so we do not need to override them as long as distance is implemented
    def compute_contrastive_loss(  # type: ignore
        self,
        features: Tensor,
        labels: Tensor,
        domains: Tensor,
        *,
        ignores: Tensor | None = None,
        mode: Literal["in-domain", "across-domain"] = "across-domain",
        temperature: float = 0.1,
        skip_background: bool = True,
        top_k: int = -1,
    ):
        # across-domain mask current domain out
        # so supports only in-domain
        return super().compute_contrastive_loss(
            features,
            labels,
            torch.zeros_like(domains),
            ignores=ignores,
            mode="in-domain",
            temperature=temperature,
            skip_background=skip_background,
            top_k=top_k)


if __name__ == "__main__":
    accelerator = Accelerator()

    num_classes = 6
    num_domains = 2
    b = 4
    dim = 128
    bank = PrototypeBank(num_domains,
                         num_classes,
                         dim,
                         accelerator=accelerator)
    bank.prototypes = torch.randn(num_domains, num_classes, dim)
    bank = accelerator.prepare(bank)

    x = torch.rand(b, dim, 16, 16, 16).to(accelerator.device)

    d1 = bank.distance(x, mode="across-domain", metric="l2")
    d2 = bank.distance(x, mode="across-domain", metric="l1")
    d3 = bank.distance(x, mode="across-domain", metric="cosine")
    d4 = bank.distance(x, mode="weighted-across-domain", metric="l2")
    d5 = bank.distance(x,
                       mode="in-domain",
                       metric="l2",
                       domains=torch.zeros(4))

    buf = []

    print_ = print

    def print(*args):
        buf.append(" ".join(map(str, args)))

    print("Distance Test:")
    print(d1.shape, d1.min(), d1.max())
    print(d2.shape, d2.min(), d2.max())
    print(d3.shape, d3.min(), d3.max())
    print(d4.shape, d4.min(), d4.max())
    print(d5.shape, d5.min(), d5.max())

    prob1 = bank.classify(x, mode="across-domain", metric="l2")
    prob2 = bank.classify(x, mode="across-domain", metric="l2", tau=0.5)
    print("Classification Test:")
    print(prob1.shape, prob2.shape)
    print(prob1[0, :, 0, 0, 0])
    print(prob2[0, :, 0, 0, 0])

    w = bank.weight_label(x,
                          torch.zeros(b, 16, 16, 16,
                                      device=accelerator.device),
                          mode="across-domain")
    print("Weight Test:")
    print(w.shape, w.min(), w.max())

    feat = torch.rand(b, dim, 4, 4, 4).to(accelerator.device)
    label = torch.randint(0, num_classes,
                          (b, 16, 16, 16)).to(accelerator.device)
    domain = torch.randint(0, 2, (b, )).to(accelerator.device)
    bank.update(feat, label, domain)

    loss = bank.compute_contrastive_loss(feat,
                                         label,
                                         domain,
                                         mode="across-domain")
    loss2 = bank.compute_contrastive_loss(feat,
                                          label,
                                          domain,
                                          mode="in-domain")
    print("Loss Test:")
    print(loss, loss2)

    bank.propagate({i: [i % num_domains] for i in range(num_classes)})

    # check prototype should be the same among all devices
    gathered = accelerator.gather(bank.prototypes)
    assert torch.allclose(*gathered.chunk(2, dim=0))

    print_("\n".join(buf))
    accelerator.end_training()
