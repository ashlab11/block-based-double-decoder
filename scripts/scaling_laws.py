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

# μP base width: HPs tuned at this width transfer to all larger widths
MUP_BASE_DIM = 64


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
mup_base_dim: {mup_base_dim}

collator_cls: "DDPretrainCollator"
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
    # μP: use a fixed base LR tuned at the smallest width
    mup_base_lr = lr_for_dim(MUP_BASE_DIM)

    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        dim, enc, dec = arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"]
        ne = non_emb_params(dim, enc, dec)
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
                lr=mup_base_lr, tokens=tokens,
                mup_base_dim=MUP_BASE_DIM,
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

    print(f"\nμP: base_dim={MUP_BASE_DIM}, base_lr={mup_base_lr}")
    print(f"  Hidden LR is auto-scaled by base_dim/dim in the optimizer.\n")
    print(f"Architectures:")
    for plabel in PARAM_LABELS:
        arch = ARCHITECTURES[plabel]
        ne = non_emb_params(arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"])
        hidden_lr = mup_base_lr * MUP_BASE_DIM / arch["dim"]
        print(f"  {plabel:>5}: dim={arch['dim']:>3}, enc={arch['num_encoder_layers']:>2}, "
              f"dec={arch['num_decoder_layers']:>2}, "
              f"non_emb={ne:>10,}, hidden_lr={hidden_lr:.1e}")


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
            print(f"[{i+1}/{total}] DRY RUN: train(params={params}, tokens={tokens}, mup_base_dim={MUP_BASE_DIM})")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name} (μP base_dim={MUP_BASE_DIM})")
        print(f"{'='*60}")

        try:
            train(params=params, tokens=tokens, mup_base_dim=MUP_BASE_DIM)
        except RuntimeError as e:
            print(f"FAILED: {name} ({e})")
            failed.append(name)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. {total - skipped - len(failed)} completed, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")


# ── Collect results ─────────────────────────────────────────────────────────

MODEL_TYPE_PREFIXES = [
    ("dd",  "Double_Decoder"),
    ("sed", "StandardEncDec"),
    ("dec", "DecoderOnly"),
]


def cmd_collect(args):
    """Read all results JSONs and output plain-text tables.

    Scans checkpoints/scaling/ for {prefix}_{plabel}_{tlabel}tok_results.json
    across all model type prefixes (dd, sed, dec) and both legacy and
    width-only param labels.
    """
    import re

    rows = []
    curves = []

    # Discover all result files by scanning the directory
    if not CHECKPOINT_DIR.exists():
        print(f"No results directory: {CHECKPOINT_DIR}")
        return

    result_files = sorted(CHECKPOINT_DIR.glob("*_results.json"))
    # Parse filenames like: dd_0.5M_10Mtok_results.json or dd_500000p_10000000tok_results.json
    pattern = re.compile(
        r"^(dd|sed|dec)_(.+?)_(\d+[A-Za-z]*)tok_results\.json$"
    )

    known_prefixes = {p: tn for p, tn in MODEL_TYPE_PREFIXES}
    seen_plabels = set()
    seen_tlabels = set()

    for fpath in result_files:
        m = pattern.match(fpath.name)
        if not m:
            # Try legacy format: dd_{plabel}_{tlabel}tok_results.json
            # where plabel might be from PARAM_LABELS
            legacy = re.match(r"^(dd)_(\d+)p_(\d+)tok_results\.json$", fpath.name)
            if legacy:
                prefix = legacy.group(1)
                # Map raw param value to label
                raw_params = int(legacy.group(2))
                raw_tokens = int(legacy.group(3))
                plabel = None
                for pl, pv in PARAM_VALUES.items():
                    if pv == raw_params:
                        plabel = pl
                        break
                tlabel = None
                for tl, tv in TOKEN_VALUES.items():
                    if tv == raw_tokens:
                        tlabel = tl
                        break
                if plabel is None or tlabel is None:
                    continue
            else:
                continue
        else:
            prefix = m.group(1)
            plabel = m.group(2)
            tlabel = m.group(3)

        if prefix not in known_prefixes:
            continue

        type_name = known_prefixes[prefix]
        seen_plabels.add(plabel)
        seen_tlabels.add(tlabel)

        # Resolve architecture for non-emb param count
        if plabel in ARCHITECTURES:
            arch = ARCHITECTURES[plabel]
        else:
            arch = None

        with open(fpath) as f:
            data = json.load(f)

        # Get params from hparams or architecture
        hparams = data.get("hparams", {})
        dim = hparams.get("dim") or (arch["dim"] if arch else 0)
        enc = hparams.get("num_encoder_layers") or (arch["num_encoder_layers"] if arch else 0)
        dec = hparams.get("num_decoder_layers") or (arch["num_decoder_layers"] if arch else 0)
        ne = non_emb_params(dim, enc, dec) if dim > 0 else data.get("non_emb_params", 0)

        tokens = hparams.get("total_tokens") or TOKEN_VALUES.get(tlabel, 0)
        flops = compute_flops(ne, tokens) if ne > 0 and tokens > 0 else 0

        name = fpath.stem.replace("_results", "")

        rows.append(dict(
            name=name, prefix=prefix, type_name=type_name,
            plabel=plabel, tlabel=tlabel,
            non_emb_params=ne, tokens=tokens, flops=flops,
            dim=dim, enc=enc, dec=dec,
            loss=data["final_eval_loss"],
            ppl=data["final_eval_ppl"],
            steps=data["total_steps"],
            total_params=data.get("total_params"),
            time_sec=data.get("training_time_sec"),
        ))

        if args.curves:
            for entry in data.get("train_curve", []):
                # Support both (step, loss) and (step, tokens_seen, loss)
                if len(entry) == 3:
                    step, tokens_at_step, loss = entry
                else:
                    step, loss = entry
                    tokens_at_step = None
                curves.append(dict(
                    name=name, prefix=prefix, non_emb_params=ne,
                    total_tokens=tokens, step=step,
                    tokens_seen=tokens_at_step, train_loss=loss,
                ))
            for entry in data.get("eval_curve", []):
                if len(entry) == 3:
                    step, tokens_at_step, loss = entry
                else:
                    step, loss = entry
                    tokens_at_step = None
                curves.append(dict(
                    name=name, prefix=prefix, non_emb_params=ne,
                    total_tokens=tokens, step=step,
                    tokens_seen=tokens_at_step, eval_loss=loss,
                ))

    # Sort labels for display
    def _sort_key(label):
        """Sort labels numerically: '0.6M' → 0.6, '14.7M' → 14.7"""
        m = re.match(r"([\d.]+)", label)
        return float(m.group(1)) if m else label

    plabels = sorted(seen_plabels, key=_sort_key)
    tlabels = sorted(seen_tlabels, key=_sort_key)

    # ── Output ──────────────────────────────────────────────────────────

    found_prefixes = sorted(set(r["prefix"] for r in rows))

    print("=== SCALING LAW RESULTS ===")
    print(f"# Model types: {', '.join(found_prefixes) or 'none'}")
    print(f"# Param widths: {', '.join(plabels)}")
    print(f"# Token budgets: {', '.join(tlabels)}")
    print(f"# Total runs found: {len(rows)}")
    print()

    # ── Summary table (one row per run) ─────────────────────────────────
    print("--- SUMMARY ---")
    cols = ["name", "model_type", "non_emb_params", "tokens", "flops", "loss", "ppl",
            "dim", "enc_layers", "dec_layers", "steps", "total_params", "time_sec"]
    print("\t".join(cols))

    for r in rows:
        vals = [
            r["name"],
            r["prefix"],
            str(r["non_emb_params"]),
            str(r["tokens"]),
            f"{r['flops']:.3e}",
            f"{r['loss']:.6f}",
            f"{r['ppl']:.4f}",
            str(r["dim"]),
            str(r["enc"]),
            str(r["dec"]),
            str(r["steps"]) if r["steps"] is not None else "N/A",
            str(r["total_params"]) if r["total_params"] is not None else "N/A",
            f"{r['time_sec']:.1f}" if r["time_sec"] is not None else "N/A",
        ]
        print("\t".join(vals))

    # ── Loss grids per model type ─────────────────────────────────────
    for prefix, type_name in MODEL_TYPE_PREFIXES:
        prefix_rows = [r for r in rows if r["prefix"] == prefix]
        if not prefix_rows:
            continue

        print()
        print(f"--- LOSS GRID: {type_name} ({prefix}) ---")
        header = f"{'params':<10}" + "".join(f"{t:>12}" for t in tlabels)
        print(header)
        print("-" * len(header))

        for plabel in plabels:
            row_str = f"{plabel:<10}"
            for tlabel in tlabels:
                match = [r for r in prefix_rows
                         if r["plabel"] == plabel and r["tlabel"] == tlabel]
                if match:
                    row_str += f"{match[0]['loss']:>12.4f}"
                else:
                    row_str += f"{'---':>12}"
            print(row_str)

    # ── FLOPs grid ────────────────────────────────────────────────────
    print()
    print("--- FLOPS GRID (params × tokens) ---")
    header = f"{'params':<10}" + "".join(f"{t:>12}" for t in tlabels)
    print(header)
    print("-" * len(header))

    for plabel in plabels:
        # Get dim from any row with this plabel
        plabel_rows = [r for r in rows if r["plabel"] == plabel]
        if not plabel_rows:
            continue
        ne = plabel_rows[0]["non_emb_params"]
        row_str = f"{plabel:<10}"
        for tlabel in tlabels:
            tlabel_rows = [r for r in rows if r["tlabel"] == tlabel]
            tokens = tlabel_rows[0]["tokens"] if tlabel_rows else 0
            flops = compute_flops(ne, tokens) if tokens > 0 else 0
            row_str += f"{flops:>12.2e}"
        print(row_str)

    # ── Training curves ────────────────────────────────────────────────
    if args.curves and curves:
        print()
        print("--- TRAINING CURVES ---")
        print("name\tmodel_type\tnon_emb_params\ttotal_tokens\tstep\ttokens_seen\ttrain_loss\teval_loss")
        from collections import defaultdict
        by_run = defaultdict(list)
        for c in curves:
            by_run[c["name"]].append(c)

        for name_key in by_run:
            for c in sorted(by_run[name_key], key=lambda x: x["step"]):
                train_l = f"{c.get('train_loss', '')}" if c.get("train_loss") is not None else ""
                eval_l = f"{c.get('eval_loss', '')}" if c.get("eval_loss") is not None else ""
                tok_seen = str(c["tokens_seen"]) if c.get("tokens_seen") is not None else ""
                print(f"{c['name']}\t{c['prefix']}\t{c['non_emb_params']}\t{c['total_tokens']}\t"
                      f"{c['step']}\t{tok_seen}\t{train_l}\t{eval_l}")


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
