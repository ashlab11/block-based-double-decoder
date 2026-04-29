#!/usr/bin/env python3
"""
Scaling law experiment runner for block-based-double-decoder.

Manages a grid of (parameter_size, token_budget) training runs and outputs
plain-text data for generating scaling law plots.

Usage:
    python scripts/scaling_laws.py generate              # Create config YAMLs
    python scripts/scaling_laws.py run [--dry-run]        # Launch training runs
    python scripts/scaling_laws.py collect                # Output plain-text results
    python scripts/scaling_laws.py collect --curves       # Include training curves
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.scaling import (
    ARCHITECTURES, PARAM_LABELS, PARAM_VALUES, TOKEN_LABELS, TOKEN_VALUES,
    SEQ_LEN, TARGET_EFFECTIVE_BATCH, TOKENS_PER_STEP,
    non_emb_params, compute_flops, lr_for_dim,
    run_name_from_labels, eval_steps_for_tokens, save_steps_for_tokens,
)
from training.api import train

CONFIG_DIR = PROJECT_ROOT / "configs" / "runs" / "scaling"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "scaling"


# ── Generate configs ────────────────────────────────────────────────────────

CONFIG_TEMPLATE = """\
# Auto-generated scaling law config: {plabel} params, {tlabel} tokens
# Non-embedding params: {non_emb:,}  |  FLOPs: {flops:.2e}
# Estimated optimizer steps: {est_steps}

model_cls: "Double_Decoder"
dim: {dim}
num_encoder_layers: {enc}
num_decoder_layers: {dec}
seq_len: {seq_len}
shared: true
logit_biases: false
init_strategy: "xavier_uniform"
gradient_checkpointing: {grad_ckpt}
use_compile: false

collator_cls: "BasicPretrainCollator"
train_file: "data/Pretrain/slimpajama_6b_packed.jsonl"
eval_file: "data/Pretrain/slimpajama_6b_eval_packed.jsonl"
tokenizer_file: "tokenizer/tokenizer_32k.json"

auto_batch_size: true
target_effective_batch: {target_eff_batch}
batch_size: 64
grad_accum_steps: 1
lr: {lr}
end_lr_ratio: 0.1
total_tokens: {tokens}

logging_steps: 4
eval_steps: {eval_steps}
save_steps: {save_steps}
output_dir: "checkpoints/scaling"
output_file_name: "{name}"

wandb_project: "dd-scaling-laws"
wandb_run_name: "{name}"
wandb_entity: "block-based-double-decoders"
"""


def cmd_generate(args):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        dim, enc, dec = arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"]
        ne = non_emb_params(dim, enc, dec)
        lr = lr_for_dim(dim)
        # Enable gradient checkpointing for larger models to save memory
        grad_ckpt = "true" if dim >= 256 else "false"

        for tlabel in TOKEN_LABELS:
            tokens = TOKEN_VALUES[tlabel]
            name = run_name_from_labels(plabel, tlabel)
            est_steps = tokens // TOKENS_PER_STEP
            flops = compute_flops(ne, tokens)

            config_text = CONFIG_TEMPLATE.format(
                plabel=plabel, tlabel=tlabel,
                non_emb=ne, flops=flops, est_steps=est_steps,
                dim=dim, enc=enc, dec=dec,
                seq_len=SEQ_LEN,
                grad_ckpt=grad_ckpt,
                target_eff_batch=TARGET_EFFECTIVE_BATCH,
                lr=lr, tokens=tokens,
                eval_steps=eval_steps_for_tokens(tokens),
                save_steps=save_steps_for_tokens(tokens),
                name=name,
            )

            path = CONFIG_DIR / f"{name}.yaml"
            path.write_text(config_text)
            count += 1

    # Print summary grid
    print(f"Generated {count} configs in {CONFIG_DIR}/\n")
    print("Grid overview (non-embedding params × tokens → est. optimizer steps):\n")
    header = f"{'params':<10}" + "".join(f"{t:>10}" for t in TOKEN_LABELS)
    print(header)
    print("-" * len(header))
    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        ne = non_emb_params(arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"])
        row = f"{plabel:<10}"
        for tlabel in TOKEN_LABELS:
            tokens = TOKEN_VALUES[tlabel]
            steps = tokens // TOKENS_PER_STEP
            row += f"{steps:>10,}"
        print(row)

    print(f"\nArchitectures:")
    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        ne = non_emb_params(arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"])
        print(f"  {plabel:>5}: dim={arch['dim']:>3}, enc={arch['num_encoder_layers']:>2}, "
              f"dec={arch['num_decoder_layers']:>2}, "
              f"non_emb={ne:>10,}, lr={lr_for_dim(arch['dim']):.1e}")


# ── Run experiments ─────────────────────────────────────────────────────────

def cmd_run(args):
    pairs = []
    for plabel in PARAM_LABELS:
        if args.only_params and plabel not in args.only_params:
            continue
        for tlabel in TOKEN_LABELS:
            if args.only_tokens and tlabel not in args.only_tokens:
                continue
            pairs.append((plabel, tlabel))

    total = len(pairs)
    skipped = 0
    failed = []

    for i, (plabel, tlabel) in enumerate(pairs):
        name = run_name_from_labels(plabel, tlabel)
        results_file = CHECKPOINT_DIR / f"{name}_results.json"

        if results_file.exists() and not args.force:
            print(f"[{i+1}/{total}] SKIP {name} (results exist, use --force to rerun)")
            skipped += 1
            continue

        params = PARAM_VALUES[plabel]
        tokens = TOKEN_VALUES[tlabel]

        if args.dry_run:
            print(f"[{i+1}/{total}] DRY RUN: train(params={params}, tokens={tokens})")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name}")
        print(f"{'='*60}")

        try:
            train(params=params, tokens=tokens)
        except RuntimeError as e:
            print(f"FAILED: {name} ({e})")
            failed.append(name)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. {total - skipped - len(failed)} completed, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")


# ── Collect results ─────────────────────────────────────────────────────────

def cmd_collect(args):
    """Read all results JSONs and output plain-text tables."""

    rows = []
    curves = []
    missing = []

    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        dim = arch["dim"]
        enc = arch["num_encoder_layers"]
        dec = arch["num_decoder_layers"]
        ne = non_emb_params(dim, enc, dec)

        for tlabel in TOKEN_LABELS:
            tokens = TOKEN_VALUES[tlabel]
            name = run_name_from_labels(plabel, tlabel)
            flops = compute_flops(ne, tokens)

            # Check both grid-style and raw-value-style result filenames
            results_file = CHECKPOINT_DIR / f"{name}_results.json"
            if not results_file.exists():
                raw_name = f"dd_{PARAM_VALUES[plabel]}p_{tokens}tok"
                results_file = CHECKPOINT_DIR / f"{raw_name}_results.json"

            if not results_file.exists():
                missing.append(name)
                rows.append(dict(
                    name=name, plabel=plabel, tlabel=tlabel,
                    non_emb_params=ne, tokens=tokens, flops=flops,
                    dim=dim, enc=enc, dec=dec,
                    loss=None, ppl=None, steps=None,
                    total_params=None, time_sec=None,
                ))
                continue

            with open(results_file) as f:
                data = json.load(f)

            rows.append(dict(
                name=name, plabel=plabel, tlabel=tlabel,
                non_emb_params=ne, tokens=tokens, flops=flops,
                dim=dim, enc=enc, dec=dec,
                loss=data["final_eval_loss"],
                ppl=data["final_eval_ppl"],
                steps=data["total_steps"],
                total_params=data.get("total_params"),
                time_sec=data.get("training_time_sec"),
            ))

            if args.curves:
                for step, loss in data.get("train_curve", []):
                    curves.append(dict(
                        name=name, non_emb_params=ne,
                        tokens=tokens, step=step, train_loss=loss,
                    ))
                for step, loss in data.get("eval_curve", []):
                    curves.append(dict(
                        name=name, non_emb_params=ne,
                        tokens=tokens, step=step, eval_loss=loss,
                    ))

    # ── Output ──────────────────────────────────────────────────────────

    print("=== SCALING LAW RESULTS ===")
    print(f"# Grid: {len(PARAM_LABELS)} param sizes x {len(TOKEN_LABELS)} token budgets "
          f"= {len(PARAM_LABELS) * len(TOKEN_LABELS)} runs")
    if missing:
        print(f"# Missing: {len(missing)} runs ({', '.join(missing)})")
    print()

    # ── Summary table (one row per run) ─────────────────────────────────
    print("--- SUMMARY ---")
    cols = ["name", "non_emb_params", "tokens", "flops", "loss", "ppl",
            "dim", "enc_layers", "dec_layers", "steps", "total_params", "time_sec"]
    print("\t".join(cols))

    for r in rows:
        vals = [
            r["name"],
            str(r["non_emb_params"]),
            str(r["tokens"]),
            f"{r['flops']:.3e}",
            f"{r['loss']:.6f}" if r["loss"] is not None else "N/A",
            f"{r['ppl']:.4f}" if r["ppl"] is not None else "N/A",
            str(r["dim"]),
            str(r["enc"]),
            str(r["dec"]),
            str(r["steps"]) if r["steps"] is not None else "N/A",
            str(r["total_params"]) if r["total_params"] is not None else "N/A",
            f"{r['time_sec']:.1f}" if r["time_sec"] is not None else "N/A",
        ]
        print("\t".join(vals))

    # ── Loss grid (for quick visual inspection) ────────────────────────
    print()
    print("--- LOSS GRID (params × tokens) ---")
    header = f"{'params':<10}" + "".join(f"{t:>12}" for t in TOKEN_LABELS)
    print(header)
    print("-" * len(header))

    for plabel in PARAM_LABELS:
        row_str = f"{plabel:<10}"
        for tlabel in TOKEN_LABELS:
            name = run_name_from_labels(plabel, tlabel)
            match = [r for r in rows if r["name"] == name]
            if match and match[0]["loss"] is not None:
                row_str += f"{match[0]['loss']:>12.4f}"
            else:
                row_str += f"{'---':>12}"
        print(row_str)

    # ── FLOPs grid ─────────────────────────────────────────────────────
    print()
    print("--- FLOPS GRID (params × tokens) ---")
    header = f"{'params':<10}" + "".join(f"{t:>12}" for t in TOKEN_LABELS)
    print(header)
    print("-" * len(header))

    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        ne = non_emb_params(arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"])
        row_str = f"{plabel:<10}"
        for tlabel in TOKEN_LABELS:
            tokens = TOKEN_VALUES[tlabel]
            flops = compute_flops(ne, tokens)
            row_str += f"{flops:>12.2e}"
        print(row_str)

    # ── Training curves ────────────────────────────────────────────────
    if args.curves and curves:
        print()
        print("--- TRAINING CURVES ---")
        print("name\tnon_emb_params\ttokens\tstep\ttrain_loss\teval_loss")
        # Group by run, interleave train/eval
        from collections import defaultdict
        by_run = defaultdict(list)
        for c in curves:
            key = c["name"]
            by_run[key].append(c)

        for name_key in by_run:
            for c in sorted(by_run[name_key], key=lambda x: x["step"]):
                train_l = f"{c.get('train_loss', '')}" if c.get("train_loss") is not None else ""
                eval_l = f"{c.get('eval_loss', '')}" if c.get("eval_loss") is not None else ""
                print(f"{c['name']}\t{c['non_emb_params']}\t{c['tokens']}\t"
                      f"{c['step']}\t{train_l}\t{eval_l}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scaling law experiments for block-based-double-decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/scaling_laws.py generate
  python scripts/scaling_laws.py run --dry-run
  python scripts/scaling_laws.py run --only-params 0.5M 2.5M
  python scripts/scaling_laws.py run --only-tokens 10M 50M
  python scripts/scaling_laws.py collect > results.tsv
  python scripts/scaling_laws.py collect --curves > results_full.tsv
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    sub.add_parser("generate", help="Create YAML configs for the full grid")

    # run
    p_run = sub.add_parser("run", help="Launch training runs")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    p_run.add_argument("--force", action="store_true",
                       help="Rerun even if results exist")
    p_run.add_argument("--only-params", nargs="+", metavar="LABEL",
                       help="Only run these param sizes (e.g., 0.5M 2.5M)")
    p_run.add_argument("--only-tokens", nargs="+", metavar="LABEL",
                       help="Only run these token budgets (e.g., 10M 50M)")

    # collect
    p_collect = sub.add_parser("collect", help="Collect results and output plain text")
    p_collect.add_argument("--curves", action="store_true",
                           help="Include per-step training curves in output")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
