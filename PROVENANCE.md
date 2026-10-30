# Source provenance

The core code was extracted from the authors' local research implementation.

| Release file | Research source |
| --- | --- |
| `cavm/model.py` | `models/swin_unetr_llama_3/swin_unetr.py` |
| `cavm/transformer.py` | `models/swin_unetr_llama_3/llama/model.py` |
| `cavm/token_swin.py` | `models/only_fce_tokentune_llama_3/tokentune/me_swin_unetr.py` |
| `cavm/memory_efficient.py` | Model construction in `models/only_fce_tokentune_llama_3/swin_unetr.py`; shared generation code is inherited from CAVM |

Packaging changes:

- Removed experiment-only profiling, unused helpers, commented debug code, and print statements.
- Removed forced CUDA allocation, global default-device mutation, and unused KV caches; the attention mask is a non-persistent buffer. Existing module parameter names are retained where applicable, but checkpoint compatibility has not been verified.
- Replaced hard-coded spatial reshape and mask lengths with values derived from `img_size`. The default dimensions reproduce the original dimensions; smaller dimensions are for the execution checks.
- Omitted the unsupported `use_v2=False` argument when constructing the MONAI SwinTransformer (the original architecture). The release pins MONAI 1.3.2 because MONAI 1.2 does not import on Python 3.12.
- Shared the CAVM generation path with ME-CAVM; only `swinViT` is replaced with the original token-split encoder. Profiling counters are omitted.
- Active token branches respect the caller's gradient mode, so outer `torch.no_grad()` works during evaluation. Frozen branches still run without gradients.
- Added synthetic examples and dependency pins. These do not reproduce dataset-specific preprocessing or the original training protocol.

Source SHA-256 hashes (for traceability, not a numerical equivalence guarantee):

The release default `prefix_ratio=0.1` was checked against the final JBHI submission (`CAVM_JBHI.pdf` and `main.tex` in `CAVM_JBHI.zip`, submitted July 2026). The Implementation Details states that all ME-CAVM experiments use `rho=0.1` except the sensitivity analysis; the Sensitivity Analysis section also identifies `rho=0.1` as the default. The synthetic Adam example uses the stated betas `(0.9, 0.99)`. These checks align the exposed settings, rather than asserting complete numerical reproducibility of the release.

- `models/swin_unetr_llama_3/swin_unetr.py`: `6153cecb7ffab160d669920dcf8b850107cbb81fc257a3e28c3f0dadc12f60b7`
- `models/swin_unetr_llama_3/llama/model.py`: `b708d12111170bc02fc797147470254b6ebf52b3ac0872a72f8126e9fb76a40e`
- `models/only_fce_tokentune_llama_3/tokentune/me_swin_unetr.py`: `f52e86abcc6925ca79264a11e3d08d97345f1510ebc69f843fc7c3b1fee7e4df`
- `models/only_fce_tokentune_llama_3/swin_unetr.py`: `53dd76a48efbd65af3d5158bdcc6f2190fd54d1ea8259d22495a53dc93bf0101`
- `models/only_fce_tokentune_llama_3/llama/model.py`: `c545b01c2ba33acfa106ed177f029c9fd7912f3c423b95fe6d3e580093b339c0`

The initial combined execution script was split into `train.py` (one synthetic feature-transformer optimizer step) and `test.py` (synthetic progressive inference), with shared options in `cavm/runtime.py`. This changes entry-point organization rather than adding a complete dataset pipeline.
