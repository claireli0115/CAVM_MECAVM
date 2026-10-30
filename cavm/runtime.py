"""Shared options for the synthetic training and inference execution checks."""

import argparse

import torch

from .model import CAVM
from .memory_efficient import MECAVM


def setup(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=["cavm", "mecavm"], default="mecavm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--feature-size", type=int, default=12)
    parser.add_argument("--prefix-ratio", type=float, default=0.1)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    kwargs = dict(
        img_size=tuple(args.size),
        in_channels=4,
        out_channels=4,
        feature_size=args.feature_size,
    )
    if args.model == "mecavm":
        kwargs["prefix_ratio"] = args.prefix_ratio
    model = (CAVM if args.model == "cavm" else MECAVM)(**kwargs).to(args.device)
    x = torch.randn(1, 4, *args.size, device=args.device)
    return model, x


def check_outputs(outputs, reference):
    assert len(outputs) == 3
    for i, output in enumerate(outputs):
        assert output.shape == reference.shape and torch.isfinite(output).all()
        print(f"Stage {i + 1}: {tuple(output.shape)}")
