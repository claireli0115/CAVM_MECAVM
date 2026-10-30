# CAVM and ME-CAVM

Core implementations of CAVM and ME-CAVM accompanying our manuscript submitted to the **IEEE Journal of Biomedical and Health Informatics (JBHI)**. The manuscript is currently under review.

## Release scope

This initial release provides the model architectures and synthetic-data training and inference entry points for the main computational paths. It includes the multi-stage feature transformer, progressive generation, and the token-split Swin encoder used by ME-CAVM.

**This is a core-method release, not a complete package for reproducing the paper's numerical results.** It does not include datasets, trained checkpoints, dataset-specific preprocessing, the original multi-stage training schedule, or evaluation scripts. The execution checks use randomly initialized weights. No patient data are included.

`train.py` performs one optimizer step using synthetic inputs and targets; `test.py` performs progressive inference with synthetic input. These entry points make the training and inference computation available for inspection and execution. Dataset loading, checkpoint-based evaluation, and the complete training schedule are not included.

## Method and code

| Component | Implementation |
| --- | --- |
| CAVM model and progressive generation | [`cavm/model.py`](cavm/model.py) |
| Staircase-masked feature transformer | [`cavm/transformer.py`](cavm/transformer.py) |
| ME-CAVM model | [`cavm/memory_efficient.py`](cavm/memory_efficient.py) |
| Token selection and active/frozen computation | [`cavm/token_swin.py`](cavm/token_swin.py) |

CAVM encodes three stages of four-channel 3D volumes. A shared feature path supplies decoder skip features; two feature transformers operate at the coarse spatial scales. Their staircase attention mask permits each stage to attend to its own and earlier stages. `forward(stage_volumes, reference_volume)` processes three supplied stages, while `test(input_volume, reference_volume)` progressively generates intermediate volumes and encodes them for subsequent stages. Both return two lists of latent features followed by a list of three output volumes.

ME-CAVM shares this generation path and replaces the FCE encoder (`swinViT`) with the token-split Swin encoder. Within each window, `prefix_ratio` controls the randomly selected active-token fraction (at least one token). Active and frozen QKV/attention/MLP paths are computed separately; the frozen paths run without autograd. Token outputs are merged back into their original positions. All tokens remain in the forward computation. The other two Swin branches retain the standard encoder. This implementation does not imply that every parameter receives gradients exclusively from active tokens; for example, output projections and residual paths remain shared.

## Setup

Use Python 3.12 in a fresh environment. Install the appropriate CPU or CUDA build of PyTorch for your machine, followed by:

```bash
python -m pip install -r requirements.txt
```

No pretrained Llama weights, text tokenizer, or dataset downloads are needed. The feature transformer operates on continuous image features.

## Quick start

Run from the repository root:

```bash
python train.py --model mecavm
python test.py --model mecavm
python train.py --model cavm
python test.py --model cavm
```

The default execution check uses a smaller `(64, 64, 64)` volume and `feature_size=12` to reduce resource requirements. It returns three tensors of shape `(1, 4, 64, 64, 64)`. This is an execution check, not a reconstruction-quality or memory benchmark. Use `--device cuda` when CUDA is available.

`train.py` freezes the image encoders and decoder and updates the two feature transformers, following the research implementation's stage-2 freeze policy. Its synthetic inputs and MSE targets are only for checking gradient flow; they are not a training dataset or the complete training protocol.

Run the small causality and token-gradient checks with `python -m unittest -v test_core`.

The research architecture defaults are `(192, 192, 128)`, `feature_size=24`, and Swin depths `(2, 2, 2, 2)`:

```python
from cavm import CAVM, MECAVM

cavm = CAVM()
mecavm = MECAVM(prefix_ratio=0.1)
```

The submitted manuscript specifies **active ratio `rho = 0.1` for all main ME-CAVM experiments**, excluding sensitivity analysis. This corresponds to `prefix_ratio=0.1` in the code and is the default in both `MECAVM()` and both entry points. The sensitivity analysis evaluates `rho` in `{1.0, 0.7, 0.5, 0.2, 0.1}`; `0.5` is a sensitivity setting rather than the main-experiment default. The same ratio is used during pretraining and finetuning.

| Parameter | Submitted manuscript / model defaults | Synthetic execution-check defaults |
| --- | --- | --- |
| Active ratio (`prefix_ratio`) | `0.1` | `0.1` |
| Input spatial size | `(192, 192, 128)` | `(64, 64, 64)` |
| Feature size | `24` (transformer dimensions `384` and `768`) | `12` (smaller execution check) |
| Batch size | `1` | `1` |

This core implementation supports batch size 1, four input/output channels, and three stages. Spatial dimensions must be multiples of 32 and at least 64; the stage-3 feature packing must also divide evenly into groups of four. The reduced execution-check size and width are not the paper's experimental configuration. Its Adam example uses `lr=1e-4` and `betas=(0.9, 0.99)` as specified in the manuscript, but does not implement the complete training protocol. Full-resolution execution can require substantial memory.

## Provenance and verification

See [`PROVENANCE.md`](PROVENANCE.md) for the source mapping and packaging changes, and [`VERIFICATION.md`](VERIFICATION.md) for the checks actually performed on this release.

## Third-party code

The token-split encoder contains MONAI-derived code under Apache-2.0. The feature transformer contains Meta Llama 2-derived code under the Llama 2 Community License. Original copyright notices are retained; see [`NOTICE`](NOTICE) and [`licenses/`](licenses/). No blanket MIT/Apache license is applied to the entire repository.
