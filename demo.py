"""Synthetic architecture demonstration; no trained predictions or patient data."""

import argparse
import torch
from cavm import CAVM, MECAVM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cavm", "mecavm"], default="mecavm")
    parser.add_argument(
        "--mode", choices=["forward", "backward", "generate"], default="forward"
    )
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
    model.train(args.mode == "backward")
    if args.mode == "backward":
        # Feature-transformer update, following the research stage-2 freeze policy.
        # This is not the paper's complete multi-stage training schedule.
        model.requires_grad_(False)
        model.model_llamalike3.requires_grad_(True)
        model.model_llamalike4.requires_grad_(True)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=1e-4, betas=(0.9, 0.99)
        )
        targets = [torch.randn_like(x) for _ in range(3)]
        _, _, outputs = model([x, x, x], x)
        loss = sum(torch.nn.functional.mse_loss(y, t) for y, t in zip(outputs, targets))
        optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
        print("Synthetic MSE:", loss.item())
    else:
        with torch.no_grad():
            _, _, outputs = (
                model.test(x, x) if args.mode == "generate" else model([x, x, x], x)
            )
    assert len(outputs) == 3
    for i, y in enumerate(outputs):
        assert y.shape == x.shape and torch.isfinite(y).all()
        print(f"Stage {i + 1}: {tuple(y.shape)}")
    print("Synthetic demo passed. Random weights; outputs have no clinical meaning.")


if __name__ == "__main__":
    main()
