"""BatchNorm statistics that are less sensitive to class-correlated batches.

`mix`: normalize with alpha * current-batch statistics + (1 - alpha) * the
supplied checkpoint's running statistics (the source-statistics prior of
Schneider et al., NeurIPS 2020).

`ema`: keep one exponential moving average of batch statistics per test-time
view, initialized from that view's first served batch and updated with
momentum alpha before normalizing, in the spirit of RoTTA's robust BN
(Yuan et al., CVPR 2023). It uses only current and previously served batches.
With `source_weight` s > 0, normalization uses (1 - s) * EMA + s * source
statistics, while the stored EMA stays a pure average of batch statistics.

`source`: normalize with the checkpoint's running statistics only (used for
selected layers in a depth-selective ablation).

Affine weights are the supplied checkpoint's and are never updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustBatchNorm2d(nn.Module):
    def __init__(self, bn, mode, alpha, source_weight=0.0):
        super().__init__()
        if mode not in ("mix", "ema", "source"):
            raise ValueError(mode)
        self.mode, self.alpha, self.eps = mode, alpha, bn.eps
        # `ema` only: blend this share of source statistics into the
        # normalization; the stored EMA itself stays a pure batch average.
        self.source_weight = source_weight
        self.register_buffer("source_mean", bn.running_mean.detach().clone())
        self.register_buffer("source_var", bn.running_var.detach().clone())
        self.weight, self.bias = bn.weight, bn.bias
        self.view = 0
        self.state = {}
        # Optional per-sample weights (summing to 1) for class-balanced batch
        # statistics; None keeps the plain batch statistics.
        self.sample_weight = None

    def forward(self, x):
        if self.mode == "source":
            # Checkpoint running statistics only; no batch statistics.
            return F.batch_norm(x, self.source_mean, self.source_var, self.weight,
                                self.bias, False, 0.0, self.eps)
        if self.sample_weight is None:
            var, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
        else:
            # Weighted batch statistics via the law of total variance.
            sample_var, sample_mean = torch.var_mean(x, dim=(2, 3), unbiased=False)
            w = self.sample_weight[:, None]
            mean = (w * sample_mean).sum(dim=0)
            var = (w * (sample_var + sample_mean * sample_mean)).sum(dim=0) - mean * mean
        if self.mode == "mix":
            a = self.alpha
            mean = a * mean + (1 - a) * self.source_mean
            var = a * var + (1 - a) * self.source_var
        else:
            previous = self.state.get(self.view)
            if previous is not None:
                a = self.alpha
                mean = (1 - a) * previous[0] + a * mean
                var = (1 - a) * previous[1] + a * var
            self.state[self.view] = (mean.detach(), var.detach())
            if self.source_weight:
                s = self.source_weight
                mean = (1 - s) * mean + s * self.source_mean
                var = (1 - s) * var + s * self.source_var
        return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0.0, self.eps)


def replace_batchnorm(model, mode, alpha, source_weight=0.0):
    """Swap every BatchNorm2d for RobustBatchNorm2d; return the new modules."""
    replaced = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                module = RobustBatchNorm2d(child, mode, alpha, source_weight)
                setattr(parent, name, module)
                replaced.append(module)
    return replaced
