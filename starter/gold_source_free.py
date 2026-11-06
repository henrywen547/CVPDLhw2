"""Source-free GOLD-inspired subspace ablation for the official HW2 stream.

This is not the released GOLD classifier: no source prototypes, EMA teacher,
source warm-up, contrastive loss, extra model, or future test data are used.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


class GoldSourceFree:
    """Online classifier-gradient subspace and one trainable scaling vector."""

    def __init__(self, model, *, lr=0.01, momentum=0.9, rank=8, alpha=0.1,
                 confidence_threshold=0.5):
        if lr <= 0 or not 0 <= momentum < 1 or not 0 < alpha <= 1:
            raise ValueError("Require lr > 0, 0 <= momentum < 1, 0 < alpha <= 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Require 0 <= confidence_threshold <= 1")
        if not isinstance(model.fc, nn.Linear):
            raise TypeError("Expected a ResNet linear classifier")
        if not 1 <= rank <= min(model.fc.in_features, model.fc.out_features):
            raise ValueError("Rank must fit the classifier row span")

        self.classifier = model.fc
        model.fc = nn.Identity()
        self.feature_extractor = model.eval()
        self.feature_extractor.requires_grad_(False)
        self.classifier.requires_grad_(False)
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

        self.rank = rank
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.coefficients = torch.ones(self.classifier.out_features,
                                       device=self.classifier.weight.device)
        self.basis = self._basis_from_coefficients()
        self.scale = nn.Parameter(torch.zeros(rank, device=self.classifier.weight.device))
        self.optimizer = torch.optim.SGD([self.scale], lr=lr, momentum=momentum)
        self.batch = 0
        self.diag_path = None
        diagnostic_output = os.environ.get("CVPDL_DIAG_JSONL")
        if diagnostic_output:
            self.diag_path = Path(diagnostic_output)
            self.diag_path.parent.mkdir(parents=True, exist_ok=True)
            self.diag_path.write_text("")

    @torch.no_grad()
    def _basis_from_coefficients(self):
        weight = self.classifier.weight.detach()
        weighted = self.coefficients.clamp_min(0).sqrt()[:, None] * weight
        gram = weighted @ weighted.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        indices = torch.argsort(eigenvalues, descending=True)[:self.rank]
        basis = weighted.T @ eigenvectors[:, indices]
        basis = basis / eigenvalues[indices].clamp_min(1e-12).sqrt()[None, :]
        old = getattr(self, "basis", None)
        if old is not None:
            sign = torch.where((basis * old).sum(0) >= 0, 1.0, -1.0)
            basis = basis * sign
        return basis

    @torch.no_grad()
    def _update_subspace(self, prediction, confidence):
        selected = prediction[confidence >= self.confidence_threshold]
        if selected.numel() < 2:
            return False
        proportions = torch.bincount(selected, minlength=self.classifier.out_features).float()
        proportions /= selected.numel()
        self.coefficients.mul_(1 - self.alpha).add_(proportions, alpha=self.alpha)
        self.basis = self._basis_from_coefficients()
        return True

    def __call__(self, x):
        # The backbone is frozen; current-batch BN statistics remain available.
        with torch.no_grad():
            features = self.feature_extractor(x)
        self.optimizer.zero_grad(set_to_none=True)
        projection = features @ self.basis
        adjusted = features + (projection * self.scale) @ self.basis.T
        logits = self.classifier(adjusted)
        prediction = logits.detach().argmax(1)
        probability = logits.detach().softmax(1)
        confidence = probability.max(1).values
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        self.optimizer.step()
        basis_updated = self._update_subspace(prediction, confidence)
        if self.diag_path is not None:
            record = dict(
                batch=self.batch,
                n=len(prediction),
                prediction=prediction.cpu().tolist(),
                confidence=confidence.cpu().tolist(),
                entropy=(-(probability * probability.clamp_min(1e-12).log()).sum(1)).cpu().tolist(),
                updated=True,
                basis_updated=basis_updated,
                scale_norm=float(self.scale.detach().norm().cpu()),
            )
            with self.diag_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
        self.batch += 1
        return prediction
