# Verification

Verified locally on 2026-10-30 with Windows, Python 3.12.14, PyTorch 2.6.0+cpu, MONAI 1.3.2, NumPy 1.26.4, and einops 0.8.1. ME-CAVM defaults were subsequently corrected to `prefix_ratio=0.1` against the final submitted manuscript, and its generation and backward demos were rerun at that ratio. The backward demo uses manuscript Adam betas `(0.9, 0.99)`.

| Command | Result |
| --- | --- |
| `python -m unittest -v test_core` | Both tests passed |
| `python demo.py --model cavm` | Three finite outputs of shape `(1, 4, 64, 64, 64)` |
| `python demo.py --model mecavm --mode generate` | Progressive generation returned three finite outputs of the same shape |
| `python demo.py --model mecavm --mode backward` | Finite loss and feature-transformer gradients; one optimizer step completed |

The unit tests check that changing the final input stage cannot alter earlier transformer outputs, and that splitting active/frozen tokens preserves attention forward values while preventing gradients to frozen input tokens. They also check that outer `no_grad` is respected.

The backward demo follows the stage-2 freeze policy: only the two feature transformers are updated. It does not test end-to-end training of every encoder/decoder parameter. The token-split encoder's gradient behavior is checked separately by the unit test.

These checks establish execution and selected computational properties for the reduced synthetic configuration. They do not validate the paper's numerical results, checkpoint compatibility, GPU execution, full-resolution memory consumption, or a complete dataset training/evaluation pipeline. MONAI emits an informational deprecation warning about `img_size`; the pinned version still supports this argument.
