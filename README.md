# block-based-double-decoder

Every script under `scripts/` is able to be ran as `bash <script>` for RunPod and as `sbatch <script>` for SLURM.

---

## CLI: RunPod

```bash
git clone https://github.com/ashlab11/block-based-double-decoder.git
cd block-based-double-decoder

bash scripts/1_setup.sh                                  # ~10-20 min, prompts wandb login
bash scripts/2_data.sh                                   # ~2-5 min (HF fast path) or ~1-2 hr (slow) if you cannot access public HF dataset
bash scripts/3_preflight.sh                              # ~3-5 min, sanity / correctness checks
bash scripts/4_micro_run.sh                              # ~1 min, dry run (micro proof of concept)
bash scripts/5_train.sh                                  # ~1-2 hr, runs one 50M training run @ 1B tokens (proof of concept)
bash scripts/6_eval.sh checkpoints/dd_50m_1btok.pt       # ~30-90 min because it's running all 22 evals
```

To resume after preemption:

```bash
bash scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
```

To run multi-GPU (although this is not optimised and only really built in incase we need it later):

```bash
NUM_GPUS=2 bash scripts/5_train.sh
```

---

## CLI: SLURM

One-time setup on the cluster (login node):

```bash
git clone https://github.com/ashlab11/block-based-double-decoder.git
cd block-based-double-decoder

module avail 2>&1 | grep -iE "conda|anaconda|miniconda|python"
module load anaconda3/2023.09-0-aqbc
conda create -n dd python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dd
cp slurm.env.example slurm.env


echo 'export WANDB_API_KEY=paste-your-key-here' >> ~/.bashrc # <-- replace paste-your-key-here
chmod 600 ~/.bashrc
source ~/.bashrc
sed -i '/^export WANDB_API_KEY=/d' slurm.env
```

Every subsequent shell session, before submitting jobs:

```bash
module load anaconda3/2023.09-0-aqbc
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

---

## Training the model in code: `train()` at arbitrary (params, tokens)

The Python entry point for one-off training runs is `training.api.train`:

```python
from training.api import train

results = train(params=15_000_000, tokens=300_000_000, mup_base_dim=64)
print(results["final_eval_loss"], results["final_eval_ppl"])
```

Thus if you want to run training across multiple different param & token counts you could simply loop through calls of train() and save the results!

If you really care, what happens under the hood is the following:

1. **Architecture interpolation** (`configs/scaling.py:interpolate_architecture`). Given `params`, picks the nearest predefined `(num_encoder_layers, num_decoder_layers)` ratio from `ARCHITECTURES` (encoder:decoder ≈ 2:1) and solves for the `dim` (rounded to the nearest multiple of 64, since `num_heads = dim // 64`) that lands closest to the target param count in log-space.
2. **Config build** (`configs/scaling.py:build_scaling_config`). Produces a fully-populated `TrainingConfig`-compatible dict: model hyperparameters, data paths, eval/save cadence (auto-tuned for ~20 eval points and ~5 checkpoints across the run), wandb metadata, and learning rate.
3. **Subprocess launch** (`training/api.py`). Invokes `torchrun --nproc_per_node=1 training/train_cli.py --params ... --tokens ... --mup-base-dim ...`.
4. **Pretrain loop** (`training/pretrain.py:pretrain`). Builds the model, sets up the optimizer (AdamW with separate parameter groups when μP is enabled), runs the polynomial-decay schedule with 5% warmup, and writes `checkpoints/scaling/<run_name>_results.json`.
5. **Return** (`training.api.train`). Reads that JSON and returns it as a dict containing `final_eval_loss`, `final_eval_ppl`, `total_steps`, `tokens_seen`, `total_params`, `training_time_sec`, `hparams`, `train_curve`, and `eval_curve`.

### The μP guarantee

Maximal Update Parameterization (μP) lets you **tune learning rate and other hyperparameters once at a small base width, then transfer them unchanged to any larger width**. Concretely, when you call `train(params=..., tokens=..., mup_base_dim=64)`:

- **Embedding output and unembedding (logit projection) are scaled by `base_dim/dim`** at every forward pass (`models/double_decoder.py:67-93`). This keeps the magnitude of activations entering the residual stream invariant in the width.
- **Attention and MLP hidden weights get a separate, smaller learning rate of `base_lr * (base_dim/dim)`** via a dedicated AdamW param group (`training/pretrain.py:331-354`). Embedding/unembedding/scalar params stay at the base LR.
- **`base_lr` itself is set by `lr_for_dim(base_dim)`** (`configs/scaling.py:lr_for_dim`) this is not tuned yet.

If you set `mup_base_dim=0` (the default), μP is disabled and the script falls back to a width-aware heuristic `lr = 2e-3 * sqrt(64/dim)`

The same logic is exposed as a CLI for shell-scripted sweeps:

```bash
torchrun --nproc_per_node=1 training/train_cli.py \
    --params 15000000 --tokens 300000000 --mup-base-dim 64
```
