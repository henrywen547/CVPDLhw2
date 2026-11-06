"""Independent, normalization-only implementation of UnMix-TNS equations.

Tomar et al., ICLR 2024, https://openreview.net/forum?id=xyxU99Nutg.
This uses only the supplied model's stored BN statistics and served unlabeled batches.
It does not include the paper's optional optimizer-based TTA combinations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnMixBN2d(nn.Module):
    """Online per-instance normalization with K source-initialized components."""

    def __init__(self, source_bn, *, components=16, alpha=0.5, tau=0.07):
        super().__init__()
        if components < 2 or not 0 < alpha < 1 or tau <= 0:
            raise ValueError("Require components >= 2, 0 < alpha < 1, tau > 0")
        if source_bn.running_mean is None or source_bn.running_var is None:
            raise ValueError("UnMix-TNS needs source checkpoint BN running statistics")
        self.eps = source_bn.eps
        self.tau = tau
        mean = source_bn.running_mean.detach()
        variance = source_bn.running_var.detach().clamp_min(source_bn.eps)
        noise = torch.randn((components, len(mean)), device=mean.device, dtype=mean.dtype)
        spread = math.sqrt(alpha * components / (components - 1))
        self.register_buffer("component_mean", mean[None] + variance.sqrt()[None] * spread * noise)
        self.register_buffer("component_var", variance[None].expand(components, -1).clone() * (1 - alpha))
        self.register_buffer("weight", source_bn.weight.detach().clone() if source_bn.affine else None)
        self.register_buffer("bias", source_bn.bias.detach().clone() if source_bn.affine else None)

    @torch.no_grad()
    def forward(self, x):
        batch_size = x.shape[0]
        instance_var, instance_mean = torch.var_mean(x, dim=(2, 3), unbiased=False)
        similarity = F.normalize(instance_mean, dim=1) @ F.normalize(self.component_mean, dim=1).T
        assignment = (similarity / self.tau).softmax(dim=1)
        probability = assignment[:, :, None]
        mixed_mean = ((1 - probability) * self.component_mean[None] +
                      probability * instance_mean[:, None])
        mixed_var = ((1 - probability) * self.component_var[None] +
                     probability * instance_var[:, None])
        center = mixed_mean.mean(dim=1)
        variance = (mixed_var + mixed_mean.square()).mean(dim=1) - center.square()
        output = (x - center[:, :, None, None]) * torch.rsqrt(variance.clamp_min(0)[:, :, None, None] + self.eps)
        if self.weight is not None:
            output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Apply this batch's statistics only to future calls, after current normalization.
        momentum = 1 - 0.9 ** (batch_size / 64)
        step = momentum / batch_size
        mass = assignment.sum(dim=0)[:, None]
        self.component_mean.add_(step * (assignment.T @ instance_mean - mass * self.component_mean))
        self.component_var.add_(step * (assignment.T @ instance_var - mass * self.component_var))
        self.component_var.clamp_(min=self.eps)
        return output


class UnMixStream:
    """Replace source BN layers and infer each served batch once."""

    def __init__(self, model):
        self.model = model.eval()
        self._replace_bn(self.model)

    @classmethod
    def _replace_bn(cls, module):
        for name, child in tuple(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, UnMixBN2d(child))
            else:
                cls._replace_bn(child)

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(dim=1)
