"""Independent LAME-inspired current-batch output correction.

Based on Boudiaf et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html.
This local variant uses cosine 5-nearest-neighbor affinity and ten fixed CCCP steps.
It only uses the current served batch and does not change model weights.
"""

import atexit
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_bn import RobustBatchNorm2d, replace_batchnorm


def knn_affinity(feature, neighbors):
    """Symmetric cosine-weighted top-k graph on L2-normalized features."""
    count = len(feature)
    similarity = feature @ feature.T
    similarity.fill_diagonal_(-1)
    values, indices = similarity.topk(min(neighbors, count - 1), dim=1)
    affinity = torch.zeros_like(similarity)
    affinity.scatter_(1, indices, values.clamp_min(0))
    return (affinity + affinity.T) * 0.5


def matched_rbf_affinity(feature, neighbors):
    """Dense RBF graph rescaled to the k-NN graph's mean row sum.

    Distances are Euclidean on L2-normalized features, as in the LAME author
    code. sigma is the batch mean of each sample's distance to its k-th
    nearest other sample. The diagonal is zero. One global scale makes the
    mean row sum equal that of `knn_affinity` on the same batch. Degenerate
    batches (zero sigma or zero RBF mass) fall back to the k-NN graph.
    """
    baseline = knn_affinity(feature, neighbors)
    count = len(feature)
    distance = torch.cdist(feature, feature)
    distance.fill_diagonal_(float("inf"))
    sigma = distance.topk(min(neighbors, count - 1), dim=1,
                          largest=False).values[:, -1].mean()
    if not torch.isfinite(sigma) or sigma <= 1e-12:
        return baseline
    rbf = torch.exp(-distance.square() / (2 * sigma * sigma))  # diagonal -> 0
    mass = rbf.sum(dim=1).mean()
    if mass <= 1e-12:
        return baseline
    return rbf * (baseline.sum(dim=1).mean() / mass)


def adjust_outputs(probability, features, *, neighbors=5, steps=10,
                   affinity="knn", stats=None):
    """Apply the paper's probability update to a symmetric feature graph.

    `stats`, if a list, receives label-free solver diagnostics per batch:
    max |z_10 - z_final|, argmax changes after step ten, and the last-step
    max residual.
    """
    count = len(probability)
    if count < 2:
        return probability
    feature = F.normalize(features, dim=1)
    if affinity == "knn":
        graph = knn_affinity(feature, neighbors)
    elif affinity == "rbf":
        graph = matched_rbf_affinity(feature, neighbors)
    else:
        raise ValueError(affinity)

    log_prior = probability.clamp_min(1e-12).log()
    assignment = probability
    at_ten = previous = None
    for step in range(steps):
        if stats is not None:
            previous = assignment
        assignment = (log_prior + graph @ assignment).softmax(dim=1)
        if stats is not None and step == 9:
            at_ten = assignment
    if stats is not None and at_ten is not None:
        stats.append(((assignment - at_ten).abs().max().item(),
                      (assignment.argmax(1) != at_ten.argmax(1)).sum().item(),
                      (assignment - previous).abs().max().item()))
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
    steps = 10
    affinity = "knn"
    solver_stats = None

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
        feature = self._graph_feature(feature)
        if self.use_memory and self.memory is not None:
            refined = adjust_outputs_with_memory(probability, feature, *self.memory,
                                                 neighbors=self.neighbors)
        else:
            refined = adjust_outputs(probability, feature, neighbors=self.neighbors,
                                     steps=self.steps, affinity=self.affinity,
                                     stats=self.solver_stats)
        if self.use_memory:
            self.memory = (feature, refined)
        self._after_batch(feature, refined)
        return refined.argmax(dim=1)

    def _prior(self, probability, feature):
        """Hook to replace the LAME prior; identity by default."""
        return probability

    def _graph_feature(self, feature):
        """Hook to replace the graph's view-averaged feature; identity."""
        return feature

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


class LameK20Ema02Src02(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.2, 0.2


class LameK20Ema03Src03(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.3


class LameK20Ema03Src04(LameEightViewK20RobustBN):
    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.4


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


class LameBalancedBN05Src03(LameBalancedBN):
    """Class-balanced statistics (gamma 0.5) plus 30% source statistics."""

    gamma = 0.5
    bn_source = 0.3


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


class LameViewTrim(LameEightViewK20RobustBN):
    """Selected src02 method with per-image view-variation trimming.

    Independent, simplified component inspired by TACT (Liu et al., *Test-Time
    Adaptation by Causal Trimming*, NeurIPS 2025). For each image, the eight
    existing view features (no extra backbone pass) are centered and the
    leading direction u of their within-image variation is found from the 8x8
    Gram matrix. Each view feature is trimmed as f - (f.u)u. With
    `trim_prior`, each trimmed view feature goes through the supplied
    classifier (bias kept) and the view softmaxes are averaged in the same
    order as the baseline. With `trim_graph`, the graph uses the mean trimmed
    feature. Only one direction is removed; images whose views carry
    (numerically) no variation are not trimmed. TACT's augmentations and
    accumulated trimmed prototypes are not used.
    """

    bn_mode, bn_alpha, bn_source = "ema", 0.3, 0.2
    trim_prior = True
    trim_graph = False
    relative_tolerance = 1e-6

    def _after_view(self, index, output):
        if index == 0:
            self.view_features = []
            self.direction = None
        self.view_features.append(self.feature)

    def _trim_direction(self):
        if self.direction is None:
            stacked = torch.stack(self.view_features, dim=1)  # (B, V, D)
            centered = stacked - stacked.mean(dim=1, keepdim=True)
            gram = centered @ centered.transpose(1, 2)  # (B, V, V)
            values, vectors = torch.linalg.eigh(gram)  # ascending
            direction = (vectors[..., -1:].transpose(1, 2) @ centered).squeeze(1)
            energy = stacked.square().sum(dim=(1, 2)) / stacked.shape[1]
            valid = values[..., -1] > self.relative_tolerance * energy
            norm = direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.direction = torch.where(valid[:, None], direction / norm,
                                         torch.zeros_like(direction))
        return self.direction

    def _trim(self, feature):
        u = self._trim_direction()
        return feature - (feature * u).sum(dim=1, keepdim=True) * u

    def _prior(self, probability, feature):
        if not self.trim_prior:
            return probability
        total = None
        for view_feature in self.view_features:
            output = self.model.fc(self._trim(view_feature)).softmax(dim=1)
            total = output if total is None else total + output
        return total / len(self.view_features)

    def _graph_feature(self, feature):
        return self._trim(feature) if self.trim_graph else feature

    def _after_batch(self, feature, refined):
        self.view_features = None
        self.direction = None


class LameTrimPrior(LameViewTrim):
    trim_prior, trim_graph = True, False


class LameTrimBoth(LameViewTrim):
    trim_prior, trim_graph = True, True


class LameSrc02Steps100(LameK20Ema03Src02):
    """Selected src02 method with 100 fixed graph updates instead of ten.

    The LAME author code allows up to 100 updates with an energy stopping
    rule; here all 100 run, avoiding per-step GPU syncs. If the environment
    variable LAME_SOLVER_DIAG names a file, label-free solver diagnostics
    (see `adjust_outputs`) are written there at exit; predictions are
    unaffected.
    """

    steps = 100

    def __init__(self, model):
        super().__init__(model)
        path = os.environ.get("LAME_SOLVER_DIAG")
        if path:
            self.solver_stats = []
            atexit.register(self._dump_stats, path)

    def _dump_stats(self, path):
        diffs, changes, residuals = zip(*self.solver_stats)
        summary = {"batches": len(diffs),
                   "max_abs_diff_step10_vs_final": max(diffs),
                   "mean_max_abs_diff_step10_vs_final": sum(diffs) / len(diffs),
                   "argmax_changes_after_step10": sum(changes),
                   "batches_with_changes": sum(c > 0 for c in changes),
                   "max_last_step_residual": max(residuals),
                   "per_batch": self.solver_stats}
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=1)


class LameSrc02Rbf(LameK20Ema03Src02):
    """Selected src02 method with the matched-strength dense RBF graph."""

    affinity = "rbf"


class LameSrc02Layer4Source(LameK20Ema03Src02):
    """Selected src02 method with checkpoint BN statistics in all of layer4.

    Depth-selective ablation inspired by the label-shift sensitivity of deep
    layers reported for Channel-Selective Normalization (CoLLAs 2024). The
    stem and layer1-layer3 keep per-view EMA 0.3 + 20% source statistics.
    Every BN in layer4, including the downsample BN, uses the checkpoint's
    running statistics. The paper's source-data channel scores are not used.
    """

    def __init__(self, model):
        super().__init__(model)
        for name, module in self.model.named_modules():
            if name.startswith("layer4.") and isinstance(module, RobustBatchNorm2d):
                module.mode = "source"


class LameEightViewMemory(LameEightView):
    """Previous served batch as fixed graph nodes; one batch of memory."""

    use_memory = True
