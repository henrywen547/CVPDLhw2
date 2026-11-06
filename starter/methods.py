"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import math

import torch
import torch.nn as nn

from diagnostics import SourceDiagnostics


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


class BatchNormEntropy:
    """One online update per served batch; return its pre-update predictions.

    This is a small independent implementation of the Tent update pattern.
    COME changes only the loss. No source data, replay, or extra weights are used.
    """

    def __init__(self, model, *, loss_name, lr, momentum, come_k):
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
        self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        self.loss_name = loss_name
        self.come_k = come_k

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
        if self.loss_name == "tent":
            loss = self.softmax_entropy(logits).mean()
        else:
            loss = self.opinion_entropy(logits).mean()
        loss.backward()
        self.optimizer.step()
        return prediction


class Tent(BatchNormEntropy):
    def __init__(self, model, *, lr=0.001, momentum=0.9, come_k=1000):
        super().__init__(model, loss_name="tent", lr=lr, momentum=momentum, come_k=come_k)


class TentCOME(BatchNormEntropy):
    def __init__(self, model, *, lr=0.001, momentum=0.9, come_k=1000):
        super().__init__(model, loss_name="come", lr=lr, momentum=momentum, come_k=come_k)


METHODS = {"source": Source, "source_diag": SourceDiagnostics, "tent": Tent, "tent_come": TentCOME}
