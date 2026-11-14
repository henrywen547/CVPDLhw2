"""Independent LAME-inspired current-batch output correction.

Based on Boudiaf et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html.
This local variant uses cosine 5-nearest-neighbor affinity and ten fixed CCCP steps.
It only uses the current served batch and does not change model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_bn import replace_batchnorm


def adjust_outputs(probability, features, *, neighbors=5, steps=10):
    """Apply the paper's probability update to a symmetric local feature graph."""
    count = len(probability)
    if count < 2:
        return probability
    feature = F.normalize(features, dim=1)
    similarity = feature @ feature.T
    similarity.fill_diagonal_(-1)
    values, indices = similarity.topk(min(neighbors, count - 1), dim=1)
    affinity = torch.zeros_like(similarity)
    affinity.scatter_(1, indices, values.clamp_min(0))
    affinity = (affinity + affinity.T) * 0.5

    log_prior = probability.clamp_min(1e-12).log()
    assignment = probability
    for _ in range(steps):
        assignment = (log_prior + affinity @ assignment).softmax(dim=1)
    return assignment


def adjust_outputs_with_memory(probability, features, memory_features,
                               memory_assignment, *, neighbors=5, steps=10):
    """Same update, with fixed nodes from the previous served batch.

    Memory nodes are already-returned outputs of the previous batch. They are
    candidate neighbors for current samples but are never updated, so earlier
    predictions cannot change.
    """
    count = len(probability)
    if count < 2:
        return probability
    feature = F.normalize(features, dim=1)
    memory = F.normalize(memory_features, dim=1)
    similarity = feature @ torch.cat((feature, memory)).T
    similarity[:, :count].fill_diagonal_(-1)
    values, indices = similarity.topk(min(neighbors, similarity.shape[1] - 1), dim=1)
    affinity = torch.zeros_like(similarity)
    affinity.scatter_(1, indices, values.clamp_min(0))
    current = (affinity[:, :count] + affinity[:, :count].T) * 0.5
    memory_message = affinity[:, count:] @ memory_assignment

    log_prior = probability.clamp_min(1e-12).log()
    assignment = probability
    for _ in range(steps):
        assignment = (log_prior + current @ assignment + memory_message).softmax(dim=1)
    return assignment


class LameCurrentBatch:
    """Current-batch BN predictions corrected by a feature affinity graph."""

    crop_borders = ()
    include_flip = False
    neighbors = 5
    use_memory = False

    def __init__(self, model):
        self.model = model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        self.feature = None
        self.memory = None
        self.model.avgpool.register_forward_hook(self._capture_feature)

    def _capture_feature(self, _module, _inputs, output):
        self.feature = output.flatten(1)

    def _select_view(self, index):
        """Hook for view-specific normalization state; no-op by default."""

    def _after_view(self, index, output):
        """Hook called with each view's softmax output; no-op by default."""

    @torch.no_grad()
    def __call__(self, x):
        views = [x]
        if self.include_flip:
            views.append(x.flip(-1))
        for border in self.crop_borders:
            crop = x[:, :, border:-border, border:-border]
            zoom = F.interpolate(crop, size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)
            views.extend((zoom, zoom.flip(-1)))

        probability = None
        feature_sum = None
        for index, view in enumerate(views):
            self._select_view(index)
            output = self.model(view).softmax(dim=1)
            self._after_view(index, output)
            probability = output if probability is None else probability + output
            feature_sum = self.feature if feature_sum is None else feature_sum + self.feature
        self.feature = None
        probability = probability / len(views)
        feature = feature_sum / len(views)
        probability = self._prior(probability, feature)
        if self.use_memory and self.memory is not None:
            refined = adjust_outputs_with_memory(probability, feature, *self.memory,
                                                 neighbors=self.neighbors)
        else:
            refined = adjust_outputs(probability, feature, neighbors=self.neighbors)
        if self.use_memory:
            self.memory = (feature, refined)
        self._after_batch(feature, refined)
        return refined.argmax(dim=1)

    def _prior(self, probability, feature):
        """Hook to replace the LAME prior; identity by default."""
        return probability

    def _after_batch(self, feature, refined):
        """Hook called after the current batch's outputs are final; no-op."""


class LameSixView(LameCurrentBatch):
    crop_borders = (8, 16)
    include_flip = True


class LameFourView(LameCurrentBatch):
    crop_borders = (8,)
    include_flip = True


class LameEightView(LameCurrentBatch):
    crop_borders = (8, 16, 24)
    include_flip = True


class LameEightViewK3(LameEightView):
    neighbors = 3


class LameEightViewK10(LameEightView):
    neighbors = 10


class LameEightViewK15(LameEightView):
    neighbors = 15


class LameEightViewK20(LameEightView):
    neighbors = 20


class LameEightViewK30(LameEightView):
    neighbors = 30


class LameEightViewK20RobustBN(LameEightViewK20):
    """k = 20 eight-view LAME with source-mixed or per-view EMA BN statistics."""

    bn_mode = "mix"
    bn_alpha = 1.0
    bn_source = 0.0

    def __init__(self, model):
        self.robust_bn = replace_batchnorm(model, self.bn_mode, self.bn_alpha,
                                           self.bn_source)
        super().__init__(model)

    def _select_view(self, index):
        for module in self.robust_bn:
            module.view = index


class LameK20Mix10(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "mix", 1.0


class LameK20Mix09(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "mix", 0.9


class LameK20Mix07(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "mix", 0.7


class LameK20Ema03(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "ema", 0.3


class LameK20Ema01(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "ema", 0.1


class LameK20Ema02(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "ema", 0.2


class LameK20Ema05(LameEightViewK20RobustBN):
    bn_mode, bn_alpha = "ema", 0.5


class LameK20Ema03Src01(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.1


class LameK20Ema03Src02(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.2


class LameK20Ema03Src03(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.3


class LameK15Ema03(LameEightViewK20RobustBN):
    neighbors = 15
    bn_mode, bn_alpha = "ema", 0.3


class LameK30Ema03(LameEightViewK20RobustBN):
    neighbors = 30
    bn_mode, bn_alpha = "ema", 0.3


class LameBalancedBN(LameEightViewK20RobustBN):
    """k = 20, per-view EMA BN 0.3, with class-balanced batch statistics.

    Simplified, independent variant of TRIBE's balanced BN (Su et al., AAAI
    2024). View 0 (the original image) is normalized with plain batch
    statistics; its argmax gives pseudo-labels for the current batch. Views
    1-7 then compute batch statistics with per-sample weights proportional to
    count(predicted class) ** -gamma, so class-correlated batches bias the
    statistics less. The weighted statistics feed the same per-view EMA.
    TRIBE's per-class running statistics and tri-net training are not used.
    """

    bn_mode, bn_alpha = "ema", 0.3
    gamma = 1.0

    def _after_view(self, index, output):
        if index != 0:
            return
        labels = output.argmax(dim=1)
        counts = torch.bincount(labels, minlength=output.shape[1]).to(output.dtype)
        weight = counts[labels].pow(-self.gamma)
        weight = weight / weight.sum()
        for module in self.robust_bn:
            module.sample_weight = weight

    def _after_batch(self, feature, refined):
        for module in self.robust_bn:
            module.sample_weight = None


class LameBalancedBN05(LameBalancedBN):
    gamma = 0.5


class LameBalancedBN10(LameBalancedBN):
    gamma = 1.0


class LameT3A(LameEightViewK20RobustBN):
    """k = 20, per-view EMA BN 0.3, with a T3A-style prototype prior.

    Independent variant of T3A (Iwasawa & Matsuo, NeurIPS 2021). The support
    set starts from the classifier's weight rows. After each batch's outputs
    are final, the batch's view-averaged features join it with LAME-refined
    labels and entropies (the original uses source-classifier argmax and
    entropy). Each class keeps its `support_size` lowest-entropy members.
    Prototype logits are `z @ normalize(sum of normalized supports per class)`,
    as in T3A; `source_share` blends the source classifier back into the prior.
    Only already-served batches enter the support set.
    """

    bn_mode, bn_alpha = "ema", 0.3
    support_size = 20
    source_share = 0.0

    def __init__(self, model):
        super().__init__(model)
        fc = self.model.fc
        weight = fc.weight.detach()
        with torch.no_grad():
            warm = fc(weight).softmax(dim=1)
        self.supports = F.normalize(weight, dim=1)
        self.labels = warm.argmax(dim=1)
        self.entropy = -(warm * warm.clamp_min(1e-12).log()).sum(dim=1)
        self.classes = fc.out_features

    def _prototype_logits(self, feature):
        onehot = F.one_hot(self.labels, self.classes).to(self.supports.dtype)
        prototypes = F.normalize(self.supports.T @ onehot, dim=0)
        return feature @ prototypes

    def _prior(self, probability, feature):
        prototype = self._prototype_logits(feature).softmax(dim=1)
        if self.source_share:
            s = self.source_share
            return s * probability + (1 - s) * prototype
        return prototype

    def _after_batch(self, feature, refined):
        entropy = -(refined * refined.clamp_min(1e-12).log()).sum(dim=1)
        supports = torch.cat((self.supports, F.normalize(feature, dim=1)))
        labels = torch.cat((self.labels, refined.argmax(dim=1)))
        entropies = torch.cat((self.entropy, entropy))
        keep = []
        for label in labels.unique():
            members = (labels == label).nonzero().flatten()
            if len(members) > self.support_size:
                members = members[entropies[members].topk(self.support_size,
                                                           largest=False).indices]
            keep.append(members)
        keep = torch.cat(keep)
        self.supports, self.labels, self.entropy = supports[keep], labels[keep], entropies[keep]


class LameT3AM20(LameT3A):
    support_size = 20


class LameT3AM100(LameT3A):
    support_size = 100


class LameT3AMixM20(LameT3A):
    support_size = 20
    source_share = 0.5


class LameEightViewMemory(LameEightView):
    """Previous served batch as fixed graph nodes; one batch of memory."""

    use_memory = True
