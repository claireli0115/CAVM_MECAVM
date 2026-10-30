"""Run progressive inference with synthetic input and randomly initialized weights.

This entry point checks inference execution; it does not evaluate reconstruction
quality on a dataset. Generated outputs have no clinical meaning.
"""

import torch

from cavm.runtime import check_outputs, setup


def main():
    model, x = setup(__doc__)
    model.eval()
    with torch.no_grad():
        _, _, outputs = model.test(x, x)
    check_outputs(outputs, x)
    print("Inference execution check passed (random weights, synthetic input).")


if __name__ == "__main__":
    main()
