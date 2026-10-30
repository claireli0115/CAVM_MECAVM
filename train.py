"""Run one feature-transformer update with synthetic inputs and targets.

This entry point checks training execution; it does not implement the complete
dataset training protocol or produce a trained research checkpoint.
"""

import torch

from cavm.runtime import check_outputs, setup


def main():
    model, x = setup(__doc__)
    model.train()
    # Follow the research stage-2 freeze policy.
    model.requires_grad_(False)
    model.model_llamalike3.requires_grad_(True)
    model.model_llamalike4.requires_grad_(True)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
        betas=(0.9, 0.99),
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
    check_outputs(outputs, x)
    print("Training execution check passed (one synthetic optimizer step).")


if __name__ == "__main__":
    main()
