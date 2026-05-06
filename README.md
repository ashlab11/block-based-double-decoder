# block-based-double-decoder

Our scaling-law experiments in this project were conducted over 3 core model architectures:

- **`dd`** — Double_Decoder (block-based double decoder, our architecture)
- **`sed`** — StandardEncDec (standard encoder–decoder)
- **`dec`** — DecoderOnlyModel (decoder-only)

The essential pipeline is 3 commands: install (`1_setup.sh`), building the dataset (`2_data.sh`), then sweeping
`(arch × tokens × model_type)` cells with `parallel_scaling.py`. Four `mup_*.py` scripts validate and
tune the per-arch base LR; `wd_sweep.py` then tunes per-arch weight decay on top of it.

---

## End-to-end RunPod recipe

```bash
git clone https://<token>@github.com/ashlab11/block-based-double-decoder.git
cd block-based-double-decoder

bash scripts/1_setup.sh                # ~10–20 min, installs torch + flash-attn
bash scripts/2_data.sh                 # ~2–5 min (HF fast path) or ~1–2 hr (slow)

wandb login <key>
hf auth login                          # paste hf_... token at the prompt

# Example of a single-cell training run with checkpoint upload to HF + wandb logging
python scripts/parallel_scaling.py \
    --arch-set large --only-arch 5M --model-types dd \
    --token-set large --token-budgets 6B \
    --batch-size 8 --grad-accum 4 \
    --save-checkpoints --checkpoint-fractions 0.5,0.9,1.0 \
    --hf-repo "bpbradle/bbdd-scaling-checkpoints" \
    --wandb-project final-sweep --wandb-entity "block-based-double-decoders" \
    --output-dir checkpoints/ben_sweep_dd_5M_6B \
    --skip-full-eval \
    2>&1 | tee logs/ben_sweep_dd_5M_6B_$(date +%Y%m%d_%H%M%S).log
```

---

## `scripts/1_setup.sh`

Pure environment setup:

1. Branches between RunPod (interactive, GPU present) and SLURM (login node
   vs. allocated job). 
2. Reads `nvidia-smi --query-gpu=compute_cap` to compute cap ≥ 10.0 (Blackwell,
   B200) and get `torch>=2.7` from the cu128 index. Everything older gets `torch==2.6.0+cu124`.
3. Installs `torchtune==0.6.0`, `torchao==0.6.1`, `transformers`, `datasets`, `hydra-core`, `omegaconf`, `matplotlib`, `tqdm`, `wandb`, `hf_transfer` and re-pins torch to prevent transitive downgrades.
4. Installs flex-attention, or atleast tries to and falls back to `flex_attention` if the compile fails.
5. Imports each package, builds a tiny `torch.compile` model, runs a forward+backward
   on the GPU, and verifies `wandb` login..

---

## `scripts/2_data.sh` 

Data follows two different paths:

- **Fast path** (default, ~2–5 min): downloads three pre-packed files from
  `bpbradle/slimpajama-6b-packed`:
  - `data/Pretrain/slimpajama_6b_packed.jsonl`
  - `data/Pretrain/slimpajama_6b_eval_packed.jsonl`
  - `tokenizer/tokenizer_32k.json`
- **Slow path** (~1–2 hr): falls back when the HF download fails. Calls
  `data/retrieval_scripts/{tokenizer_corpus,slimpajama,pack_dataset}.py` to build everything from
  scratch.

After either path, **step 2d** unconditionally tokenizes UltraChat (~50M tokens, ~5 min) into
`data/SFT/ultrachat.jsonl` + `ultrachat_eval.jsonl`. This runs even when you don't plan to use
`--run-sft`, because `parallel_scaling.py`'s SFT path expects those files to already exist.


To verify the fast-path download one can run:

```bash
md5sum data/Pretrain/slimpajama_6b_eval_packed.jsonl tokenizer/tokenizer_32k.json
# 07750348c40097f3f14cf1ed448c3dc6  data/Pretrain/slimpajama_6b_eval_packed.jsonl
# a1e47e4340b20676a6abf02a572d593e  tokenizer/tokenizer_32k.json
```

---

## `scripts/parallel_scaling.py`

This is the main endpoint we used and allows for sweeping through one or more `(arch × tokens × model_type)` cells. Models are torch-compiled once and
re-initialized for each token budget. Per-cell results land as JSON in `--output-dir` with checkpoints
optionally uploaded to HF Hub.

### Grid axes

| Axis | Flag | Default | Values |
| --- | --- | --- | --- |
| Architecture set | `--arch-set {small,large}` | `small` | `small` = 0.6M–28.9M (dim 64–448); `large` = 5M–300M (dim 192–1408). 8 enc + 4 dec layers in both. |
| Token set | `--token-set {small,large}` | `small` | `small` = 10M–600M; `large` = 100M–6B. |
| Filter archs | `--only-arch 5M,25M` | all | Comma-separated labels from the chosen arch set. |
| Filter tokens | `--token-budgets 100M,1B` | all | Comma-separated labels. |
| Model types | `--model-types dd,dec` | `dd,sed,dec` | Pick any subset. |

### Common invocations

**Print the grid + per-cell FLOPs and exit (no training):**
```bash
python scripts/parallel_scaling.py --arch-set large --token-set large --dry-run
```

**Small sweep:**
```bash
python scripts/parallel_scaling.py --arch-set small --token-set small
```

**Split the large sweep across GPUs:**
```bash
# L40 tier
python scripts/parallel_scaling.py --arch-set large --token-set large --only-arch 5M,25M
# H100 tier
python scripts/parallel_scaling.py --arch-set large --token-set large --only-arch 50M,150M
# B200 tier
python scripts/parallel_scaling.py --arch-set large --token-set large --only-arch 300M
```

All three write into the same `--output-dir`, producing a unified result set.

**One arch, one token budget, all three model types — useful single-cell debug:**
```bash
python scripts/parallel_scaling.py --arch-set large --only-arch 50M --token-budgets 100M
```

**Cheap PPL only — skip the benchmark suite entirely:**
```bash
python scripts/parallel_scaling.py --arch-set small --skip-full-eval
```

**Full smoke test — pretrain + paper_full evals + SFT + paper_full evals (post-SFT):**
```bash
python scripts/parallel_scaling.py \
    --arch-set large --only-arch 5M --token-set large --token-budgets 100M \
    --auto-batch-size \
    --eval-suite paper_full --eval-max-examples 50 \
    --run-sft --sft-tokens 50000000
```

**Mid-training loss curve (5 intermediate eval points):**
```bash
python scripts/parallel_scaling.py --arch-set small --mid-eval-points 5
```

**Save checkpoints at 50%, 90%, 100% and push every checkpoint to HF Hub:**
```bash
python scripts/parallel_scaling.py --arch-set large --only-arch 5M --token-budgets 6B \
    --save-checkpoints --checkpoint-fractions 0.5,0.9,1.0 \
    --hf-repo bpbradle/bbdd-scaling-checkpoints
```

**Disable μP and run as a vanilla transformer at fixed LR:**
```bash
python scripts/parallel_scaling.py --no-mup --peak-lr 3e-4 --arch-set large --only-arch 50M
```

**Boundary ablation (DD only — `sed`/`dec` ignore the flag):**
```bash
python scripts/parallel_scaling.py --boundary-strategy single_middle ...
# choices: random_uniform (default) | prompt_style | single_middle | logspace
```

### Full flag reference

**Batch / step budget**
- `--batch-size N` (default 16) — per-step batch.
- `--grad-accum N` (default 32) — accumulation steps.
- `--auto-batch-size` — probe the largest fitting batch per `(arch, gpu)`; derive `grad_accum` from
  `--target-effective-batch`. Overrides the two flags above.
- `--target-effective-batch N` — target effective batch in sequences. Default AUTO scales with model
  size (96 for 5M up to 512 for 300M), then capped per cell so each cell runs for at least
  `--min-optimizer-steps` (default 500).
- `--max-batch-size N` (default 128) — ceiling for the auto-tune search.

**Optimizer / μP**
- `--no-mup` — disable μP. Models are built with `mup_base_dim=0`, all param groups use the base LR
  directly. Use only with `--peak-lr` at a width-appropriate value (e.g. `3e-4` for `dim=576`).
- `--peak-lr LR` — override pretrain base LR. Bypasses `configs/mup_tuned.json`. Does not affect
  SFT (use `--sft-lr`).

**Eval**
- `--eval-batches N` (default 10) — forward batches for the cheap end-of-training PPL eval.
- `--mid-eval-points N` (default 0) — number of intermediate PPL eval points; 5 gives a usable loss
  curve.
- `--eval-suite paper|paper_full|quick|all|intrinsic|<csv>` (default `paper`) — which benchmarks
  run after training.
- `--eval-max-examples N` (default 500) — per-eval example cap.
- `--skip-full-eval` — skip the benchmark suite entirely (only the cheap PPL runs).
- `--eval-data-file PATH` — held-out file for intrinsic ppl/bpb (default: derived from `--eval-file`).

**SFT step (off by default)**
- `--run-sft` / `--no-sft` — run UltraChat SFT after pretrain + pretrain-eval, then re-run the eval
  suite. The per-run JSON gains `pretrain_evals` + `sft_evals` keys.
- `--sft-tokens N` (default 50M).
- `--sft-train-file PATH`, `--sft-eval-file PATH` — default to `data/SFT/ultrachat{,_eval}.jsonl`.
- `--sft-lr LR` (default 2e-5), `--sft-grad-accum N` (default 4).

**I/O**
- `--output-dir DIR` (default `checkpoints/parallel_scaling`).
- `--save-checkpoints` — write `<output-dir>/<model_type>_<arch>_<tokens>tok.pt` per cell.
- `--checkpoint-fractions 0.5,0.9,1.0` — save at intermediate progress fractions; each produces a
  `_pct{NNN}.pt` file. Requires `--save-checkpoints`.
- `--hf-repo USER/REPO` — upload each saved checkpoint to HF Hub. Path-in-repo:
  `<model_type>/<arch>/<tokens>tok/pct{NNN}.pt`.
- `--hf-private` — make the HF repo private if it doesn't exist yet.
- `--no-compile` — disable `torch.compile`.
- `--train-file`, `--eval-file`, `--tokenizer-file` — override the default Pretrain paths.

**Wandb (opt-in)**
- `--wandb-project NAME` — enables wandb logging. One run per `(arch, tokens, model_type)` cell.
- `--wandb-entity NAME` — team/user.
- `--wandb-run-name-prefix STR` — namespacing for related sweeps.

**Misc**
- `--dry-run` — plan only; print resolved batch sizes and FLOPs/cell.
- `--boundary-strategy` — DD-only block-boundary distribution.

---

## μP and weight-decay tuning

μP (Maximal Update Parameterization) lets you tune learning rate once at a small base width and
transfer it unchanged to any larger width. The four `mup_*.py` scripts validate that μP is
implemented correctly and tune the per-arch base LR that `parallel_scaling.py` reads from
`configs/mup_tuned.json`. Once base LR is locked, `wd_sweep.py` tunes per-arch weight decay at
the same base width and writes `configs/wd_tuned.json`, which `parallel_scaling.py` also reads
at startup.

Recommended sequence (run rarely; outputs are small JSON files committed to the repo):

```text
mup_full_check.py       # verify implementation correctness (~1 min)
mup_base_sweep.py       # tune base LR per arch — writes configs/mup_tuned.json
mup_verdict.py          # render the transfer-quality figure + classify
wd_sweep.py             # tune WD per arch at the tuned LR — writes configs/wd_tuned.json
parallel_scaling.py …   # the actual scaling-law sweep
```

### `mup_verify.py` — fast correctness check

This builds Double_Decoder at multiple widths, runs a few forward+backward steps, and
checks that (1) activation norms stay stable across widths, (2) parameter update norms scale
correctly (hidden ∝ `base_dim/dim`, embed ∝ 1), (3) at the base width, μP reduces to SP. Also
exposes a slow `--full` mode that runs an actual LR transfer experiment across widths.

```bash
python scripts/mup_verify.py                    # ~30s coord check (works on CPU)
python scripts/mup_verify.py --full --tokens 50000000   # hours, GPU
python scripts/mup_verify.py --plot             # replot from previous --full results
```

What's customizable:

| Flag | Default | Effect |
| --- | --- | --- |
| `--check` | on (default) | Fast coord check. |
| `--full` | off | Run the LR transfer experiment. Slow. |
| `--plot` | off | Replot from `checkpoints/mup_verify/results.json`. |
| `--tokens` | 50M | Token budget per `--full` run. |
| `--params` | `500K,2.5M,5M,15M` | Comma-separated param counts to probe. |
| `--lrs` | `3e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2` | LR grid. |
| `--mode {sp,mup,both}` | `both` | Whether to run standard parameterization, μP, or both. |
| `--device` | auto | `cuda` if available, else `cpu`. |

### `mup_full_check.py` — comprehensive μP verification (7 ordered checks)

What it does: probes the implementation in seven layers, in increasing order of cost:

1. Raw-signal coord check at init
2. Coord check after K=5 Adam steps (feature-learning regime)
3. Update-RMS scaling check (Adam: embed flat, hidden ∝ `base_dim/dim`)
4. Checks the sanity of the base-dim (μP at base width should be ≡ SP)
5. Checks the sanity of optimizer param-group numbers
6. Checks the logit magnitude across widths
7. **(only with `--full`)** LR-transfer experiment — hours on GPU

```bash
python scripts/mup_full_check.py                # checks 1–6, ~1 min
python scripts/mup_full_check.py --full         # all 7, hours
python scripts/mup_full_check.py --only 1,3,5   # subset
```

What's customizable:

| Flag | Effect |
| --- | --- |
| `--full` | Append check 7 (LR transfer). |
| `--only 1,3,5` | Run only listed checks. |
| `--device cuda\|cpu` | Override device. |
| `--params`, `--lrs`, `--tokens` | Inputs to check 7 only. |

Constants like `BASE_DIM=64`, `WIDTHS=[64,128,256,512,1024]`, `SEQ_LEN=256`, `BATCH=4`, and the
pass/warn thresholds (`FLAT_RATIO_OK=2.0`, `FLAT_RATIO_WARN=5.0`, etc.) are defined at the top of
the file — edit there, no CLI flag.

### `mup_base_sweep.py` — tune per-arch base LR

What it does: runs the full LR grid at every width, multi-seed, with a mid-training coord check at
25%/50%/75% of training to catch residual-stream / attention drift that the static checks in
`mup_full_check.py` can't see. 

Output:
- `checkpoints/mup_base_sweep/results.json` — every `(arch, dim, lr, seed)` cell
- `configs/mup_tuned.json` — per-arch base LR (argmin at base width, seed-mean) — **this is what
  `parallel_scaling.py` reads at startup**

```bash
python scripts/mup_base_sweep.py --archs dd,sed,dec
python scripts/mup_base_sweep.py --archs dd --lrs 1e-3,2e-3,4e-3 --seeds 0
python scripts/mup_base_sweep.py --plot-only        # only re-render mup_verdict.png
python scripts/mup_base_sweep.py --dry-run          # plan only
```

What's customizable:

| Flag | Default | Effect |
| --- | --- | --- |
| `--archs` | `dd,sed,dec` | Which model types to sweep. |
| `--dims` | `64,128,256` | Widths to sweep. The argmin at the smallest width defines the "base LR" written to `configs/mup_tuned.json`. |
| `--lrs` | `3e-4,6e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2` | LR grid. |
| `--seeds` | `0,1` | Multi-seed for noise estimation. |
| `--tokens` | 100M | Token budget per cell (doubled vs. legacy Phase 1). |
| `--enc`, `--dec` | 4, 2 | Layer counts (kept fixed across the sweep). |
| `--no-compile` | off | Skip `torch.compile`. |
| `--plot-only` | off | Re-run `mup_verdict.py` from existing `results.json`. |
| `--dry-run` | off | Print the plan and exit. |

### `mup_verdict.py` — render the transfer-quality figure

Reads `mup_base_sweep`'s `results.json` and produces a 4-panel figure per arch:

1. Base-width LR sweep (seed-mean ± seed-range)
2. Full LR×width grid (seed-mean curves; stars at argmin per width)
3. Argmin-LR vs width on log-log axes — the verdict plot
4. Mid-training coord-check trajectory at each width's empirical-best LR

It also classifies the sweep:

| Verdict | Meaning |
| --- | --- |
| `PASS` | Every width's seed-mean argmin lands on the same LR. |
| `PASS*` | Argmins span 1 grid step but not monotonic — within resolution. |
| `WARN` | Argmins drift monotonically by 1 grid step — try a wider/finer grid. |
| `FAIL` | Spread > 2× — μP transfer is genuinely broken. |

```bash
python scripts/mup_verdict.py
python scripts/mup_verdict.py --results path/to/results.json --out path/to/figure.png
```

### `wd_sweep.py` — tune per-arch weight decay

For each arch in `{dd, sed, dec}` it holds the μP-tuned base LR fixed (read from
`configs/mup_tuned.json`) and sweeps weight decay over a small log-spaced grid at the μP base
width (`dim=64`, ~0.5M params, 50M tokens — same cell shape as one `mup_base_sweep` cell, so
wall-clock is predictable: ~1 hr for 5 WDs × 3 archs × 2 seeds on one GPU). The argmin per arch
(seed-mean) lands in `configs/wd_tuned.json`, which `parallel_scaling.py` reads at startup to
override the per-arch WD.

Output:
- `checkpoints/wd_sweep/results.json` — every `(arch, wd, seed)` cell
- `configs/wd_tuned.json` — per-arch best WD, format:
  `{"dd": {"weight_decay": 0.5, "best_loss": 5.54, "tuned_lr_used": 0.015, "wd_curve": [...]}, ...}`

```bash
# Default: 5 WDs × 3 archs × 2 seeds = 30 cells, ~1 hr on one GPU
python scripts/wd_sweep.py --archs dd,sed,dec

# Quick smoke test (single arch, single seed, 10M tokens, custom grid)
python scripts/wd_sweep.py --archs dd --wds 0.0,0.05,0.1 --seeds 0 --tokens 10000000

# Plan only — print the cell list and exit
python scripts/wd_sweep.py --dry-run

# Replot from existing results.json (no training)
python scripts/wd_sweep.py --plot-only
```

What's customizable:

| Flag | Default | Effect |
| --- | --- | --- |
| `--archs` | `dd,sed,dec` | Which model types to sweep. |
| `--wds` | `0.0,0.01,0.05,0.1,0.2` | Log-spaced WD grid centered on the AdamW default. |
| `--seeds` | `0,1` | Multi-seed for noise estimation. The argmin uses seed-mean loss. |
| `--tokens` | 50M | Tokens per cell. Matches one `mup_base_sweep` Phase-1 cell so wall-clock is comparable. |
| `--dim` | 64 | Sweep width. Default = `MUP_BASE_DIM`; tuning at the base width is the whole μP-style argument. |
| `--enc`, `--dec` | 4, 2 | Layer counts (kept fixed across the sweep). |
| `--no-compile` | off | Disable `torch.compile`. |
| `--dry-run` | off | Print the plan and exit. |
| `--plot-only` | off | Re-render plots from `checkpoints/wd_sweep/results.json` only. |

---

## Outputs

| Path | Producer | Contents |
| --- | --- | --- |
| `checkpoints/<output-dir>/<arch>_<tokens>_<mt>_results.json` | `parallel_scaling.py` | One per cell: hparams, train/eval curves, optional `pretrain_evals` / `sft_evals`. |
| `checkpoints/<output-dir>/<mt>_<arch>_<tokens>tok[_pctNNN].pt` | `parallel_scaling.py --save-checkpoints` | Model weights. |
| HF Hub: `<mt>/<arch>/<tokens>tok/pct{NNN}.pt` | `parallel_scaling.py --hf-repo` | Same checkpoints, uploaded. |
| `checkpoints/mup_base_sweep/results.json` | `mup_base_sweep.py` | Full grid for the verdict figure. |
| `configs/mup_tuned.json` | `mup_base_sweep.py` | Per-arch base LR — read by `parallel_scaling.py` at startup. |
| `mup_base_sweep/mup_verdict.png` | `mup_verdict.py` | 4-panel transfer-quality figure. |
| `checkpoints/wd_sweep/results.json` | `wd_sweep.py` | Full `(arch, wd, seed)` grid + per-cell loss curve. |
| `configs/wd_tuned.json` | `wd_sweep.py` | Per-arch tuned weight decay — read by `parallel_scaling.py` at startup. |
| `logs/<job>-<id>.out` | SLURM | Job stdout/stderr. |
