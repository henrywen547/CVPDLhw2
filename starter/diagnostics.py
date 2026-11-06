"""Optional validation diagnostics from batches served by the official harness.

No labels, image IDs, or files under data/ are accessed here.
"""

import json
import os
from pathlib import Path

import torch


class SourceDiagnostics:
    """Same predictions as Source, with per-image prediction and input summaries."""

    def __init__(self, model):
        self.model = model.eval()
        output = os.environ.get("CVPDL_DIAG_JSONL")
        if not output:
            raise ValueError("Set CVPDL_DIAG_JSONL for source_diag")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("")
        self.batch = 0

    @torch.no_grad()
    def __call__(self, x):
        logits = self.model(x)
        probability = logits.softmax(dim=1)
        prediction = logits.argmax(dim=1)
        confidence = probability.max(dim=1).values
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
        # These are measurements in normalized tensor space, not raw RGB.
        intensity = x.mean(dim=(1, 2, 3))
        contrast = x.flatten(1).std(dim=1, unbiased=False)
        edge = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        record = dict(
            batch=self.batch,
            n=x.shape[0],
            prediction=prediction.cpu().tolist(),
            confidence=confidence.cpu().tolist(),
            entropy=entropy.cpu().tolist(),
            intensity=intensity.cpu().tolist(),
            contrast=contrast.cpu().tolist(),
            horizontal_difference=edge.cpu().tolist(),
        )
        with self.output.open("a") as file:
            file.write(json.dumps(record) + "\n")
        self.batch += 1
        return prediction
