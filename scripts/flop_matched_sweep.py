#!/usr/bin/env python3
"""Run Chinchilla-preserving FLOP-matched comparison points.

Item #1 of the paper-prep plan. The main parallel_scaling.py sweep trains all
three archs at iso-input-tokens, which gives DD ~1.33× and SED ~0.94× the
FLOPs of DEC at each grid point. A reviewer will rightly ask whether DD's
gain is from architecture or just more compute. This script generates the
answer: for each DEC grid point (N_dec, T_dec), it computes the FLOP-matched
(N_x, T_x) for X ∈ {dd, sed} that

    (a) matches DEC's total training FLOPs exactly, and
    (b) keeps T_x = 20 · N_x  (i.e. stays Chinchilla-optimal).

Both (params, tokens) shrink for DD (×0.866) and grow modestly for SED
(×1.032). The non-DEC archs are then trained at those interpolated sizes,
with `flop_matched_to: dec_<size>_<tokens>tok` recorded in the per-run JSON
so plots can pair them up post-hoc.

Implementation: rather than re-launch the full parallel_scaling pipeline per
point (compile overhead would dominate), we shell out to `train()` from
training.api which spawns one torchrun per run. That keeps each run cheap
to plan and easy to retry, and it reuses the existing μP / config plumbing.

NOTE: training.api currently routes to Double_Decoder only. Until SED/DEC
support lands in train_cli.py, this script produces only DD FLOP-matched
points. SED FLOP-matched points are recorded as a `pending.jsonl` queue that
parallel_scaling.py can consume directly via --params/--tokens overrides.

Usage:
    # Plan-only: see what (N, T) points it would launch
    python scripts/flop_matched_sweep.py --dry-run

    # Run DD FLOP-matched points against the dec/large reference grid
    python scripts/flop_matched_sweep.py --reference-arch dec \\
        --reference-arch-set large --reference-token-set large \\
        --source-archs dd

    # Subset
    python scripts/flop_matched_sweep.py --token-budgets 100M,500M --source-archs dd
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.scaling import (chinchilla_flop_match, interpolate_architecture,
                             non_emb_params, ARCH_SETS, TOKEN_SETS)


def _planned_points(reference_arch, ref_arch_set, ref_token_set, source_archs,
                    only_archs=None, only_tokens=None,
                    chinchilla_ratio=20.0):
    """Build the list of (source_arch, source_n, source_t, ref_label) triples
    that the sweep should run. Returns a list of dicts ready for serialization."""
    points = []
    for size_label, arch_cfg in ARCH_SETS[ref_arch_set]:
        if only_archs and size_label not in only_archs:
            continue
        ref_n = non_emb_params(arch_cfg["dim"],
                               arch_cfg["num_encoder_layers"],
                               arch_cfg["num_decoder_layers"])
        for tok_label, tok_count in TOKEN_SETS[ref_token_set]:
            if only_tokens and tok_label not in only_tokens:
                continue
            for src in source_archs:
                m = chinchilla_flop_match(
                    target_n=ref_n, target_tokens=tok_count,
                    target_arch=reference_arch, source_arch=src,
                    chinchilla_ratio=chinchilla_ratio,
                    enc=arch_cfg["num_encoder_layers"],
                    dec=arch_cfg["num_decoder_layers"])
                # Snap requested params to a valid arch (rounded multiple-of-64
                # dim). This will introduce a small (< few %) FLOP-match error
                # which we report so callers can see it.
                snapped = interpolate_architecture(m["non_emb_params"])
                points.append({
                    "ref_arch": reference_arch,
                    "ref_size_label": size_label,
                    "ref_token_label": tok_label,
                    "ref_non_emb_params": ref_n,
                    "ref_tokens": tok_count,
                    "source_arch": src,
                    "requested_n": m["non_emb_params"],
                    "requested_t": m["tokens"],
                    "snapped_n": snapped["actual_non_emb_params"],
                    "snapped_dim": snapped["dim"],
                    "snapped_enc": snapped["num_encoder_layers"],
                    "snapped_dec": snapped["num_decoder_layers"],
                    "tokens": m["tokens"],  # token count is exact (no snap)
                    "target_flops": m["target_flops"],
                    "source_flops_after_snap":
                        # k_source · 6 · N_snap · T_request (T not snapped)
                        m["source_flops"]
                        * (snapped["actual_non_emb_params"] / m["non_emb_params"]),
                    "tag": f"flopmatch_{src}_to_{reference_arch}_"
                           f"{size_label}_{tok_label}tok",
                })
    return points


def _print_plan(points):
    print(f"\n{'─'*100}")
    print(f"  FLOP-matched plan: {len(points)} runs")
    print(f"{'─'*100}")
    print(f"{'reference':<24}  {'src':<4}  {'req_N':>11}  {'snap_N':>11}  "
          f"{'tokens':>13}  {'target_FLOPs':>13}  {'snap_FLOPs':>13}  ratio")
    for p in points:
        ratio = p["source_flops_after_snap"] / p["target_flops"]
        ref = f"{p['ref_arch']}_{p['ref_size_label']}_{p['ref_token_label']}tok"
        print(f"  {ref:<24}  {p['source_arch']:<4}  "
              f"{p['requested_n']:>11,}  {p['snapped_n']:>11,}  "
              f"{p['tokens']:>13,}  {p['target_flops']:>13.3e}  "
              f"{p['source_flops_after_snap']:>13.3e}  {ratio:.3f}×")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-arch", default="dec",
                        choices=["dec", "dd", "sed"],
                        help="Architecture whose iso-token grid defines the "
                             "FLOP budget (default: dec — the cheapest baseline)")
    parser.add_argument("--reference-arch-set", default="large",
                        choices=sorted(ARCH_SETS.keys()))
    parser.add_argument("--reference-token-set", default="large",
                        choices=sorted(TOKEN_SETS.keys()))
    parser.add_argument("--source-archs", default="dd",
                        help="Comma-separated archs to FLOP-match against the "
                             "reference (default: dd; add sed if/when train_cli "
                             "supports it)")
    parser.add_argument("--only-arch", default=None,
                        help="Restrict to specific reference size labels (e.g. 5M,25M)")
    parser.add_argument("--token-budgets", default=None,
                        help="Restrict to specific reference token labels (e.g. 100M)")
    parser.add_argument("--chinchilla-ratio", type=float, default=20.0,
                        help="T:N ratio held during the FLOP match (default 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan only, don't launch any training")
    parser.add_argument("--output-dir", default="checkpoints/flop_matched")
    parser.add_argument("--mup-base-dim", type=int, default=64)
    parser.add_argument("--no-wandb", action="store_true", default=True)
    args = parser.parse_args()

    source_archs = [s.strip() for s in args.source_archs.split(",")]
    only_archs = (set(s.strip() for s in args.only_arch.split(","))
                  if args.only_arch else None)
    only_tokens = (set(s.strip() for s in args.token_budgets.split(","))
                   if args.token_budgets else None)

    points = _planned_points(
        reference_arch=args.reference_arch,
        ref_arch_set=args.reference_arch_set,
        ref_token_set=args.reference_token_set,
        source_archs=source_archs,
        only_archs=only_archs,
        only_tokens=only_tokens,
        chinchilla_ratio=args.chinchilla_ratio)
    _print_plan(points)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "flop_matched_plan.json"
    plan_path.write_text(json.dumps(points, indent=2))
    print(f"\nPlan written to: {plan_path}")

    # Queue of (arch, params, tokens) for sed (since training.api currently
    # only handles DD). parallel_scaling.py can consume this via a future
    # --extra-points flag, or you can launch each manually with:
    #   python scripts/parallel_scaling.py --model-types sed \\
    #       --arch-set large --only-arch <closest> --token-budgets <closest>
    pending_path = out_dir / "pending_sed.jsonl"
    pending = [p for p in points if p["source_arch"] != "dd"]
    if pending:
        with pending_path.open("w") as f:
            for p in pending:
                f.write(json.dumps(p) + "\n")
        print(f"Queued {len(pending)} non-DD points to: {pending_path}")
        print(f"  (These need to be launched via parallel_scaling.py once "
              f"train_cli supports SED arbitrary-N runs.)")

    if args.dry_run:
        print("\n[DRY RUN] Not launching training.")
        return

    # Launch DD FLOP-matched runs (the only ones training.api currently knows
    # how to dispatch — it builds a Double_Decoder via configs.scaling).
    from training.api import train

    dd_points = [p for p in points if p["source_arch"] == "dd"]
    print(f"\nLaunching {len(dd_points)} DD FLOP-matched runs sequentially "
          f"(each spawns its own torchrun)...")
    sweep_t0 = time.time()
    for i, p in enumerate(dd_points, 1):
        t0 = time.time()
        run_name = p["tag"]
        print(f"\n[{i}/{len(dd_points)}] {run_name}: "
              f"N={p['snapped_n']:,}  T={p['tokens']:,}")
        try:
            r = train(
                params=p["snapped_n"], tokens=p["tokens"],
                mup_base_dim=args.mup_base_dim, lr=None,
                run_name=run_name, no_wandb=args.no_wandb)
            r["flop_matched_to"] = (f"{p['ref_arch']}_{p['ref_size_label']}"
                                    f"_{p['ref_token_label']}tok")
            r["flop_matched_meta"] = p
            # Re-write augmented JSON. training.api's checkpoint path is fixed.
            results_path = (PROJECT_ROOT / "checkpoints" / "scaling"
                            / f"{run_name}_results.json")
            if results_path.exists():
                results_path.write_text(json.dumps(r, indent=2))
                print(f"  done in {(time.time() - t0):.0f}s "
                      f"-> {results_path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nFLOP-matched sweep finished in {(time.time() - sweep_t0)/60:.1f} min.")
    print(f"Pair these with the iso-tokens runs in checkpoints/parallel_scaling/")
    print(f"to make the apples-to-apples comparison plot.")


if __name__ == "__main__":
    main()
