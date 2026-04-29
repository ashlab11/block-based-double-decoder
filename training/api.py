"""Public API for training with arbitrary param/token counts.

Usage:
    from training.api import train
    results = train(params=1000000, tokens=40000000)
"""

import json
import os
import subprocess
from pathlib import Path

from configs.scaling import build_scaling_config, run_name_from_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train(params, tokens):
    """Train a model and return the results dict.

    Builds a config for the given (params, tokens) pair via architecture
    interpolation, launches training as a torchrun subprocess, and returns
    the results JSON written by pretrain().

    Args:
        params: target non-embedding parameter count (int)
        tokens: total training tokens (int)

    Returns:
        dict with final_eval_loss, final_eval_ppl, total_steps, tokens_seen,
        total_params, training_time_sec, hparams, train_curve, eval_curve.

    Raises:
        RuntimeError: if the training subprocess exits with a non-zero code.
    """
    name = run_name_from_values(params, tokens)
    results_path = PROJECT_ROOT / "checkpoints" / "scaling" / f"{name}_results.json"

    cmd = [
        "torchrun", "--nproc_per_node=1",
        "training/train_cli.py",
        f"--params={params}",
        f"--tokens={tokens}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {name} (exit code {result.returncode})")

    with open(results_path) as f:
        return json.load(f)
