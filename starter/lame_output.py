"""Independent LAME-inspired current-batch output correction.

Based on Boudiaf et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html.
This local variant uses cosine 5-nearest-neighbor affinity and ten fixed CCCP steps.
It only uses the current served batch and does not change model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LameCurrentBatch:
    """Current-batch BN predictions corrected by a feature affinity graph."""

    crop_borders = ()
    include_flip = False

    def __init__(self, model):
        self.model = model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        self.feature = None
        self.model.avgpool.register_forward_hook(self._capture_feature)

    def _capture_feature(self, _module, _inputs, output):
        self.feature = output.flatten(1)

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
        for view in views:
            output = self.model(view).softmax(dim=1)
            probability = output if probability is None else probability + output
            feature_sum = self.feature if feature_sum is None else feature_sum + self.feature
        self.feature = None
        probability = probability / len(views)
        refined = adjust_outputs(probability, feature_sum / len(views))
        return refined.argmax(dim=1)


class LameSixView(LameCurrentBatch):
    crop_borders = (8, 16)
    include_flip = True


class LameFourView(LameCurrentBatch):
    crop_borders = (8,)
    include_flip = True


class LameEightView(LameCurrentBatch):
    crop_borders = (8, 16, 24)
    include_flip = True
