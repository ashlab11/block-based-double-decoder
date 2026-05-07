#!/usr/bin/env python3
"""Fast pretraining sweep for B200 — curated 4-cell subset of parallel_scaling.

Trains exactly the cells needed for the post-hoc prefixLM cross-architecture
comparison story, no more:

    Cell                  Arch label    Tokens    Model types
    ───────────────────   ──────────    ──────    ───────────
    100M @ 2B             100M          2B        dd, sed
    50M  @ 2B             50M           2B        sed, dec

Total: 4 pretraining runs (dd_100M_2B, sed_100M_2B, sed_50M_2B, dec_50M_2B).

After this finishes, the post-hoc prefixLM pass picks them up automatically:
    python scripts/post_hoc_prefixlm.py --skip-existing --wandb-project sft

B200-tuned defaults baked in:
  - --auto-batch-size + --max-batch-size 256 lets the largest fitting batch
    absorb HBM3e headroom (a 192 GB card eats much bigger micro-batches
    than the H100/L40 default heuristic assumes).
  - 2B token budget across all four cells keeps wallclock predictable.
  - --skip-full-eval drops the benchmark suite — the headline comparison
    number comes from post_hoc_prefixlm's held-out loss anyway.

Each cell is run as a separate parallel_scaling.py subprocess (grouped by
arch since --model-types is multi-valued). Output streams live to stdout
and is also tee'd to logs/sweep_fast_<label>_<timestamp>.log.

Each pretrain checkpoint is uploaded to HF immediately after it completes,
so a B200 dying after even one cell preserves that work.

Env overrides (all optional):
    HF_REPO        (default: bpbradle/bbdd-scaling-checkpoints)
    WANDB_ENTITY   (default: block-based-double-decoders)
    WANDB_PROJECT  (default: final-sweep)

Usage:
    bash scripts/1_setup.sh
    bash scripts/2_data.sh
    hf auth login
    python scripts/parallel_scaling_fast.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (arch_label, comma-separated model_types). Token budget is 2B for all cells
# in this sweep — change here if you want a different budget.
CELLS = [
    ("100M", "dd,sed"),    # 2 runs: dd_100M_2Btok, sed_100M_2Btok
    ("50M",  "sed,dec"),   # 2 runs: sed_50M_2Btok, dec_50M_2Btok
]
TOK_BUDGET = "2B"

HF_REPO = os.environ.get("HF_REPO", "bpbradle/bbdd-scaling-checkpoints")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "block-based-double-decoders")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "final-sweep")


def run_cell(arch_label, model_types):
    """Invoke parallel_scaling.py for one (arch, model_types) group.

    Streams subprocess output live to stdout AND tees it to a log file so the
    user can watch progress in tmux while still having a persistent log.
    """
    label = f"{model_types.replace(',', '+')}_{arch_label}_{TOK_BUDGET}"
    log_path = PROJECT_ROOT / "logs" / (
        f"sweep_fast_{label}_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / "checkpoints" / f"sweep_fast_{label}"

    print(f"\n{'═'*70}", flush=True)
    print(f"  Training cell: {label}", flush=True)
    print(f"  Output dir:    {output_dir}", flush=True)
    print(f"  Log:           {log_path}", flush=True)
    print(f"{'═'*70}", flush=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "parallel_scaling.py"),
        "--arch-set", "large",
        "--only-arch", arch_label,
        "--model-types", model_types,
        "--token-set", "large",
        "--token-budgets", TOK_BUDGET,
        "--auto-batch-size",
        "--max-batch-size", "256",
        "--save-checkpoints",
        "--checkpoint-fractions", "1.0",
        "--hf-repo", HF_REPO,
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--output-dir", str(output_dir),
        "--skip-full-eval",
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    # Stream subprocess output line-by-line to both stdout and the log file.
    # bufsize=1 + text=True gives line-buffered text mode so each printed
    # line appears immediately rather than being batched at process exit.
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"\n  ✗ FAILED for {label} (exit {proc.returncode})", flush=True)
        return False
    print(f"\n  ✓ done: {label}", flush=True)
    return True


def main():
    print(f"Fast sweep starting at {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  HF_REPO:       {HF_REPO}")
    print(f"  WANDB_ENTITY:  {WANDB_ENTITY}")
    print(f"  WANDB_PROJECT: {WANDB_PROJECT}")
    print(f"  Cells: {CELLS}")
    print(f"  Tok budget per cell: {TOK_BUDGET}")
    print(f"  Total runs: {sum(len(mts.split(',')) for _, mts in CELLS)}")

    n_ok = n_fail = 0
    failed_cells = []
    for arch, mts in CELLS:
        if run_cell(arch, mts):
            n_ok += 1
        else:
            n_fail += 1
            failed_cells.append(f"{mts}@{arch}")

    print(f"\n{'═'*70}", flush=True)
    print(f"  Summary: {n_ok}/{len(CELLS)} cells succeeded, {n_fail} failed", flush=True)
    if failed_cells:
        print(f"  Failed: {failed_cells}", flush=True)
    print(f"{'═'*70}", flush=True)
    print(f"  Next step — equalize the 4 archs on prefixLM and compare:", flush=True)
    print(f"    python scripts/post_hoc_prefixlm.py --skip-existing --wandb-project sft",
          flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
