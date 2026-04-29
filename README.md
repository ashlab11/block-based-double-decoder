# block-based-double-decoder

CSCI 2951N final paper + workshop. Block-based encoder–decoder pretraining at the 50M–1B parameter scale.

The same scripts run on **RunPod** (interactive containers) and **SLURM** (HPC clusters). Pick one section below.

---

## CLI: RunPod

Cold-start on a fresh container, single H100/H200 pod:

```bash
git clone https://github.com/ashlab11/block-based-double-decoder.git
cd block-based-double-decoder
git checkout ben

bash scripts/1_setup.sh                                  # ~10-20 min, prompts wandb login
bash scripts/2_data.sh                                   # ~2-5 min (HF fast path) or ~1-2 hr (slow)
bash scripts/3_preflight.sh                              # ~3-5 min, sanity checks
bash scripts/4_micro_run.sh                              # ~1 min, dry run
bash scripts/5_train.sh                                  # ~1-2 hr, full 50M @ 1B tokens
bash scripts/6_eval.sh checkpoints/dd_50m_1btok.pt       # ~30-90 min, all 22 evals
```

To resume after preemption:

```bash
bash scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
```

To run multi-GPU (e.g. 2 GPUs):

```bash
NUM_GPUS=2 bash scripts/5_train.sh
```

---

## CLI: SLURM

One-time setup on the cluster (login node):

```bash
git clone https://github.com/ashlab11/block-based-double-decoder.git
cd block-based-double-decoder
git checkout ben

# Create and activate a Python env (required — the scripts refuse to install
# into the system python on a SLURM cluster).
conda create -n dd python=3.11 -y
conda activate dd

# Copy the env-var template and edit DATA_DIR / CKPT_DIR / WANDB_API_KEY
# to match your cluster.
cp slurm.env.example slurm.env
${EDITOR:-nano} slurm.env
```

Every subsequent shell session, before submitting jobs:

```bash
conda activate dd
source slurm.env
```

Submit the pipeline:

```bash
sbatch scripts/1_setup.sh                                     # GPU job, ~20 min
sbatch scripts/2_data.sh                                      # CPU job, ~5 min - 2 hr
sbatch scripts/3_preflight.sh                                 # GPU job, ~5 min
sbatch scripts/4_micro_run.sh                                 # GPU job, ~3 min
sbatch scripts/5_train.sh                                     # GPU job, ~1-2 hr
sbatch scripts/6_eval.sh checkpoints/dd_50m_1btok.pt          # GPU job, ~30-90 min
```

Job logs land in `logs/<job-name>-<jobid>.out`. Check status with `squeue -u $USER`.

To resume training after preemption (the job auto-resubmits itself on SIGUSR1, but you can also do it manually):

```bash
sbatch scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
```

To chain jobs so each one starts after the previous succeeds:

```bash
JID1=$(sbatch --parsable scripts/1_setup.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/2_data.sh)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 scripts/3_preflight.sh)
JID4=$(sbatch --parsable --dependency=afterok:$JID3 scripts/4_micro_run.sh)
sbatch         --dependency=afterok:$JID4 scripts/5_train.sh
```

---

## Per-script reference

Every script under `scripts/` is dual-mode: `bash <script>` on RunPod, `sbatch <script>` on SLURM. The `#SBATCH` lines at the top of each file are inert comments under `bash` and resource directives under `sbatch`.

### `scripts/1_setup.sh` — install + verify
Pins PyTorch 2.6.0+cu124, installs torchtune/torchao/transformers/datasets/wandb/hf_transfer, and best-effort builds flash-attn. Verifies CUDA is visible, runs a `torch.compile` smoke test, and confirms wandb auth.

On SLURM, refuses to run if there's no GPU visible (login node), no active conda env, or no `WANDB_API_KEY`. On RunPod, falls through to interactive `wandb login` if no key is set.

### `scripts/2_data.sh` — data pipeline
Tries the fast path first: pulls pre-packed 6B-token SlimPajama JSONL + 32K BPE tokenizer from `bpbradle/slimpajama-6b-packed` on HuggingFace (~2-5 min). Falls back to the slow path (build tokenizer → download raw text → tokenize → pack, ~1-2 hr) if the fast path fails.

Honours `DATA_DIR` and `TOKENIZER_DIR` env vars (set in `slurm.env`) by symlinking them into the repo-relative `data/Pretrain/` and `tokenizer/` paths the rest of the pipeline expects. CPU + network only; no GPU needed.

### `scripts/3_preflight.sh` — sanity checks
Runs `tests/preflight.sh`: 11 checks including a 100-step micro-train, a memory profile, a torch.compile compatibility test, and a DDP smoke test (skipped if only 1 GPU). Catches OOMs, kernel mismatches, and config drift before you burn money on a long run.

### `scripts/4_micro_run.sh` — micro dry run
Trains the real 50M model for 50 steps using `configs/runs/pretrain_50m_micro.yaml`. Validates that wandb logging, checkpointing, and the eval loop all work end-to-end. Look for loss starting near 10.4 and decreasing, no NaN values.

### `scripts/5_train.sh` — full pretrain
Launches `training/pretrain.py` with `configs/runs/pretrain_50m.yaml`: ~54M-param Double Decoder, 1B tokens (Chinchilla-optimal), seq_len 2048, bf16 autocast, auto batch size sized to fit GPU memory.

On SLURM the launcher becomes `srun torchrun ...` and the script installs a SIGUSR1 trap (90 s before the time limit) that resubmits itself with `--resume <latest-checkpoint>`, so jobs survive preemption without manual intervention. Honours `CKPT_DIR` for scratch storage.

### `scripts/6_eval.sh` — full eval suite
Runs all 22 evals (LAMBADA, HellaSwag, ARC, PIQA, SQuAD, XSum, TriviaQA, … plus the encoder-advantaging tasks for SFT'd models) on a given checkpoint. Optional second arg caps examples per eval for fast debugging.

### `scripts/6_eval_test.sh` — eval smoke test
One example per eval — confirms every eval's code path runs without errors. Use after editing `evals/`.

### `scripts/run_comparison.sh` / `scripts/run_sft_and_eval.sh` / `scripts/run_all.sh`
Higher-level orchestrators that chain the above for the three-way architecture comparison (Decoder-Only vs Double Decoder vs Std Enc-Dec) at 50M params, run SFT on UltraChat, then re-eval. Designed for RunPod-style interactive use; on SLURM, run the underlying steps directly via the dependency-chain pattern in the SLURM CLI section above.

### `scripts/scaling_laws.py`
Manages the (parameter, token) grid for scaling-law experiments. `generate` writes per-cell config YAMLs; `run` launches them; `collect` dumps results. Uses `training.api.train` under the hood.

### `scripts/mup_verify.py`
Diagnostic that confirms μP is wired correctly: trains a small grid of widths at fixed HPs and checks that loss curves overlap (the empirical μP signature).

---

## Training the model in code: `train()` at arbitrary (params, tokens) with μP

The Python entry point for one-off training runs is `training.api.train`:

```python
from training.api import train

results = train(params=15_000_000, tokens=300_000_000, mup_base_dim=64)
print(results["final_eval_loss"], results["final_eval_ppl"])
```

What happens under the hood:

1. **Architecture interpolation** (`configs/scaling.py:interpolate_architecture`). Given `params`, picks the nearest predefined `(num_encoder_layers, num_decoder_layers)` ratio from `ARCHITECTURES` (encoder:decoder ≈ 2:1) and solves for the `dim` (rounded to the nearest multiple of 64, since `num_heads = dim // 64`) that lands closest to the target param count in log-space. This avoids re-deriving an architecture per scale and keeps the layer-ratio constant across the grid.
2. **Config build** (`configs/scaling.py:build_scaling_config`). Produces a fully-populated `TrainingConfig`-compatible dict: model hyperparameters, data paths, eval/save cadence (auto-tuned for ~20 eval points and ~5 checkpoints across the run), wandb metadata, and learning rate.
3. **Subprocess launch** (`training/api.py`). Invokes `torchrun --nproc_per_node=1 training/train_cli.py --params ... --tokens ... --mup-base-dim ...`. The subprocess boundary is what makes this safe to call from notebooks or sweep drivers — each run gets its own CUDA context.
4. **Pretrain loop** (`training/pretrain.py:pretrain`). Builds the model, sets up the optimizer (AdamW with separate parameter groups when μP is enabled), runs the polynomial-decay schedule with 5% warmup, and writes `checkpoints/scaling/<run_name>_results.json`.
5. **Return** (`training.api.train`). Reads that JSON and returns it as a dict containing `final_eval_loss`, `final_eval_ppl`, `total_steps`, `tokens_seen`, `total_params`, `training_time_sec`, `hparams`, `train_curve`, and `eval_curve`.

### The μP guarantee

Maximal Update Parameterization (μP) lets you **tune learning rate and other hyperparameters once at a small base width, then transfer them unchanged to any larger width**. Concretely, when you call `train(params=..., tokens=..., mup_base_dim=64)`:

- **Embedding output and unembedding (logit projection) are scaled by `base_dim/dim`** at every forward pass (`models/double_decoder.py:67-93`). This keeps the magnitude of activations entering the residual stream invariant in the width.
- **Attention and MLP hidden weights get a separate, smaller learning rate of `base_lr * (base_dim/dim)`** via a dedicated AdamW param group (`training/pretrain.py:331-354`). Embedding/unembedding/scalar params stay at the base LR.
- **`base_lr` itself is set by `lr_for_dim(base_dim)`** (`configs/scaling.py:lr_for_dim`) — i.e. the LR that was chosen to be optimal at the base width. With μP, you do *not* re-tune this when scaling up.

The practical consequence: if a hyperparameter sweep at `mup_base_dim=64` finds an optimal LR, you can call `train(params=1_000_000_000, tokens=20_000_000_000, mup_base_dim=64)` and use that same LR with no re-tuning. The internal LR scaling for hidden weights is automatic.

If you set `mup_base_dim=0` (the default), μP is disabled and the script falls back to a width-aware heuristic `lr = 2e-3 * sqrt(64/dim)` — fine for one-off training, but does **not** guarantee HP transfer across widths.

### Common shapes

```python
# Tiny run, 0.5M params, 10M tokens — fast iteration on a single H100
train(params=500_000, tokens=10_000_000, mup_base_dim=64)

# Mid-size, with explicit LR override (e.g. from a μP sweep)
train(params=15_000_000, tokens=300_000_000, mup_base_dim=64, lr=2e-3)

# Plain (non-μP) run with auto LR
train(params=5_000_000, tokens=50_000_000)

# Custom run name (default is "dd_<params>p_<tokens>tok")
train(params=2_500_000, tokens=50_000_000, mup_base_dim=64, run_name="ablation_v3")
```

### CLI equivalents

The same logic is exposed as a CLI for shell-scripted sweeps:

```bash
torchrun --nproc_per_node=1 training/train_cli.py \
    --params 15000000 --tokens 300000000 --mup-base-dim 64
```

For grid sweeps with structured logging across (params × tokens) cells, use `scripts/scaling_laws.py` rather than calling `train()` in a loop — it handles config materialisation, deduplication of completed runs, and result collection.
