"""Run a TTA method on the val or test stream with the official harness.

  python run.py --method source --split val      # labeled: prints accuracy
  python run.py --method source --split test     # writes out/test_pred.csv for Kaggle
"""
import argparse
import random

import numpy as np
import torch
import torchvision

from harness import run_stream
from methods import METHODS


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = getattr(torchvision.models, ck["arch"])(num_classes=len(ck["wnids"]))
    model.load_state_dict(ck["state_dict"])
    return model.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="source", choices=sorted(METHODS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="source_resnet50_160.pth")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta-lr", type=float, default=0.001)
    ap.add_argument("--tta-momentum", type=float, default=0.9)
    ap.add_argument("--come-k", type=float, default=1000.0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = load_model(args.ckpt)
    if args.method in ("tent", "tent_come"):
        method = METHODS[args.method](model, lr=args.tta_lr, momentum=args.tta_momentum,
                                      come_k=args.come_k)
    else:
        method = METHODS[args.method](model)
    run_stream(method, args.data, args.split, args.out, extra_log=vars(args))


if __name__ == "__main__":
    main()
