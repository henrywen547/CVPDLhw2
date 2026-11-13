"""BatchNorm statistics that are less sensitive to class-correlated batches.

`mix`: normalize with alpha * current-batch statistics + (1 - alpha) * the
supplied checkpoint's running statistics (the source-statistics prior of
Schneider et al., NeurIPS 2020).

`ema`: keep one exponential moving average of batch statistics per test-time
view, initialized from that view's first served batch and updated with
momentum alpha before normalizing, in the spirit of RoTTA's robust BN
(Yuan et al., CVPR 2023). It uses only current and previously served batches.

Affine weights are the supplied checkpoint's and are never updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustBatchNorm2d(nn.Module):
    def __init__(self, bn, mode, alpha):
        super().__init__()
        if mode not in ("mix", "ema"):
            raise ValueError(mode)
        self.mode, self.alpha, self.eps = mode, alpha, bn.eps
        self.register_buffer("source_mean", bn.running_mean.detach().clone())
        self.register_buffer("source_var", bn.running_var.detach().clone())
        self.weight, self.bias = bn.weight, bn.bias
        self.view = 0
        self.state = {}

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
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
        return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0.0, self.eps)


def replace_batchnorm(model, mode, alpha):
    """Swap every BatchNorm2d for RobustBatchNorm2d; return the new modules."""
    replaced = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                module = RobustBatchNorm2d(child, mode, alpha)
                setattr(parent, name, module)
                replaced.append(module)
    return replaced
