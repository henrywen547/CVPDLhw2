"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import math
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostics import SourceDiagnostics
from gold_source_free import GoldSourceFree


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


class BatchStats:
    """Use only current-batch BN statistics without parameter updates."""

    def __init__(self, model):
        self.model = model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


class BatchStatsFlip(BatchStats):
    """Average current-batch BN predictions from original and horizontal-flip views."""

    @torch.no_grad()
    def __call__(self, x):
        original = self.model(x).softmax(1)
        flipped = self.model(x.flip(-1)).softmax(1)
        return (original + flipped).argmax(1)


class BatchStatsFlipZoom(BatchStats):
    """Three-view ensemble with a fixed 90% center crop resized to input size."""

    @torch.no_grad()
    def __call__(self, x):
        original = self.model(x).softmax(1)
        flipped = self.model(x.flip(-1)).softmax(1)
        border = max(1, round(min(x.shape[-2:]) * 0.05))
        crop = x[:, :, border:-border, border:-border]
        zoomed = F.interpolate(crop, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        zoom = self.model(zoomed).softmax(1)
        return (original + flipped + zoom).argmax(1)


class BatchNormEntropy:
    """One online update per served batch; return its pre-update predictions.

    This is a small independent implementation of the Tent update pattern.
    COME changes only the loss. No source data, replay, or extra weights are used.
    """

    def __init__(self, model, *, loss_name, lr, momentum, come_k,
                 max_class_share_for_update=None):
        if lr <= 0 or not 0 <= momentum < 1 or come_k <= 0:
            raise ValueError("Require lr > 0, 0 <= momentum < 1, come_k > 0")
        self.model = model.train()
        self.model.requires_grad_(False)
        parameters = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.affine:
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)
                    parameters.extend((module.weight, module.bias))
        if not parameters:
            raise ValueError("No BatchNorm2d affine parameters found")
        self.adapt_parameters = parameters
        self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        self.loss_name = loss_name
        self.come_k = come_k
        self.max_class_share_for_update = max_class_share_for_update
        self.diag_path = None
        diagnostic_output = os.environ.get("CVPDL_DIAG_JSONL")
        if diagnostic_output:
            self.diag_path = Path(diagnostic_output)
            self.diag_path.parent.mkdir(parents=True, exist_ok=True)
            self.diag_path.write_text("")
        self.batch = 0

    @staticmethod
    def softmax_entropy(logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

    def opinion_entropy(self, logits):
        # COME's norm operation is the identity in value and changes the gradient.
        # log_softmax over C+1 outcomes avoids exp(logits) overflow.
        norm = logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        opinion_logits = logits / norm * norm.detach()
        unknown = opinion_logits.new_full((len(logits), 1), math.log(self.come_k))
        log_probability = torch.cat((opinion_logits, unknown), dim=1).log_softmax(1)
        probability = log_probability.exp()
        return -(probability * log_probability).sum(1)

    @torch.enable_grad()
    def __call__(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        prediction = logits.detach().argmax(1)
        update = True
        if self.max_class_share_for_update is not None:
            counts = prediction.bincount(minlength=logits.shape[1])
            update = bool(counts.max().item() / len(prediction) <= self.max_class_share_for_update)
        before = ([parameter.detach().clone() for parameter in self.adapt_parameters]
                  if update and self.diag_path is not None else None)
        if update:
            if self.loss_name == "tent":
                loss = self.softmax_entropy(logits).mean()
            else:
                loss = self.opinion_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
        if self.diag_path is not None:
            update_l2 = (float(torch.sqrt(sum((parameter.detach() - old).square().sum()
                                              for parameter, old in zip(self.adapt_parameters, before))).item())
                         if before is not None else 0.0)
            probability = logits.detach().softmax(1)
            record = dict(
                batch=self.batch,
                n=len(prediction),
                prediction=prediction.cpu().tolist(),
                confidence=probability.max(1).values.cpu().tolist(),
                entropy=(-(probability * probability.clamp_min(1e-12).log()).sum(1)).cpu().tolist(),
                updated=update,
                update_l2=update_l2,
            )
            with self.diag_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
        self.batch += 1
        return prediction


class Tent(BatchNormEntropy):
    def __init__(self, model, *, lr=0.001, momentum=0.9, come_k=1000):
        super().__init__(model, loss_name="tent", lr=lr, momentum=momentum, come_k=come_k)


class TentPost(Tent):
    """Update on the served batch, then predict that same batch once more."""

    def __call__(self, x):
        super().__call__(x)
        with torch.no_grad():
            return self.model(x).argmax(1)


class TentCOME(BatchNormEntropy):
    def __init__(self, model, *, lr=0.001, momentum=0.9, come_k=1000):
        super().__init__(model, loss_name="come", lr=lr, momentum=momentum, come_k=come_k)


class TentGate(BatchNormEntropy):
    """Custom P2 ablation: skip updates for concentrated current predictions."""

    def __init__(self, model, *, lr=0.001, momentum=0.9, come_k=1000, max_class_share=0.4):
        if not 0 < max_class_share <= 1:
            raise ValueError("Require 0 < max_class_share <= 1")
        super().__init__(model, loss_name="tent", lr=lr, momentum=momentum, come_k=come_k,
                         max_class_share_for_update=max_class_share)


METHODS = {"source": Source, "source_diag": SourceDiagnostics, "bn_batch": BatchStats,
           "bn_flip": BatchStatsFlip, "bn_flip_zoom": BatchStatsFlipZoom,
           "tent": Tent, "tent_post": TentPost,
           "tent_come": TentCOME, "tent_gate": TentGate, "gold_sf": GoldSourceFree}
