# DeepSeek V4 gfx1250 hybrid decode

Opt-in regional fusion for DeepSeek-V4-Flash on one gfx1250 (TP1) and
DeepSeek-V4-Pro on four gfx1250 GPUs (TP4). This follows the post-load installer,
deployment refusals, lazy binding, TP agreement, per-width preparation and native
fallback used by `deepseek_v41/mono` and `kimi_k3/mono`. The original gfx950
FlyDSL kernels and their dispatch are unchanged.

This is the regional hybrid implementation developed in the prior experiments, packaged for
ATOM's mono integration. It does not claim a new full-layer kernel or new
performance results from the integration alone.

## Activate

The environment must include AITER gfx1250 Gluon FP8 preshuffle GEMM and the
public `aiter.ops.triton.fusions.mhc.mhc_post_pre_quant_gfx1250` wrapper. The
latter is supplied by `scripts/patches/dsv4_gfx1250_mhc_quant.patch`, originally
validated against AITER `7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe`.
Apply that patch to a compatible AITER checkout before installing ATOM. If the
public wrapper is already present, do not apply it twice. An unavailable kernel
refuses hybrid binding on every TP rank and logs the original exception; the
native model remains available.

```bash
pip install -e .
bash scripts/serve_dsv4_mono_gfx1250.sh flash
# or, with four free gfx1250 GPUs:
bash scripts/serve_dsv4_mono_gfx1250.sh pro
```

`MODEL_PATH`, `HIP_VISIBLE_DEVICES`, and `PORT` override launcher defaults.
The launcher selects its adjacent checkout automatically; `ATOM_SOURCE_ROOT`
can override it. A copy under `/app/scripts` also recognizes the isolated
`/opt/atom-gfx1250-integration` installation.
For an existing launch command, set `ATOM_DSV4_MONO=1` and `ATOM_MONO_ENABLE=1`.
The first flag defaults to **0**. Set either flag to 0 for the native baseline.
Use a fresh ATOM compilation cache after source changes; preserve old cache
folders instead of overwriting experimental evidence.

`ATOM_MONO_CHECK=1` compares each selected regional output exactly to the native
operation, up to 16 checks per weight/width. Check-mode comparisons run outside
capture; use `--level 0 --enforce-eager` to check actual serving tensors on every
step. CUDA graph replays do not execute Python check code.

## Regions and fallback

| Profile | Eligible decode rows | Replacements |
|---|---|---|
| Flash TP1 | 1–16 | mHC reduce/apply + RMSNorm epilogue also emits FP8 input and per-128 E8M0 scales; separate tuned attention and shared-expert Gluon GEMMs at 1/8 rows |
| Pro TP4 | 1–64 | same mHC/quant fusion; separate unsplit QKV Gluon GEMM through 64 rows; measured multicast projection configurations at 64 rows |

A tuning table selects validated `(M,N,K)` combinations. Missing keys use
native GEMM. Linear bias, output padding, and row-parallel all-reduce follow the
native order. Attention compression, indexer and shared-expert streams retain
their native ordering. All routed MoE remains native. The earlier experimental
Flash C1 routed-MoE fusion was slower in its serving ablation and is deliberately
not enabled by this integration; its source and results remain in the experiment
handoff.

Prefill, batches above the profile limit (including C128), other geometries or
architectures, EP/DP/PP/CP, TBO, plugins, speculative decoding, rollout weight
updates and incompatible compilation modes retain native dispatch. KV must be
BF16, index cache FP8, and FP8 weights use the measured preshuffle/E8M0 ABI.

The installer runs after loading and before graph warmup. It wraps the outer
model; it does not replace the compiled native inner model. The runner binds
only after paged caches have final addresses. Module views share loaded GPU
weights and cache tensors while owning their selected linear children; native
children are not modified. Hash-routing token IDs are published before each
hybrid forward. Failed binding is agreed across TP ranks; a failed width is
refused independently. Warmup uses public quantization to create the actual
column-major-byte scale ABI before capture. There are no new inter-GPU
mailboxes or persistent synchronization grids in this recommended hybrid.

Pro integration validation found a one-element BF16 mismatch at C8 with the
earlier split-K QKV configuration. The integration uses unsplit QKV throughout;
older Pro benchmark numbers must not be presented as measurements of this revised
configuration. `ATOM_MONO_FAILURE_DIR` optionally records failing projection
inputs for offline reproduction.

## Tests

```bash
python -m pytest tests/models/deepseek_v4_mono -q
```

The CPU tests cover deployment and row refusals, disabled lazy imports, native
fallback, logits delegation, hash IDs, shared weight ownership and the linear
bias/padding/TP-reduction ABI. Serving validation must additionally confirm the
`DSV4 gfx1250 ... hybrid prepared` log, exact checks on actual decode tensors,
and CUDA-graph requests at both eligible and fallback widths. For performance,
run two full clients for each configuration and report only the second client.
