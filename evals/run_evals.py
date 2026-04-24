#!/usr/bin/env python
"""
CLI entry point for running all evaluations on a trained model.

Usage:
    python evals/run_evals.py --checkpoint checkpoints/model.pt --evals all
    python evals/run_evals.py --checkpoint checkpoints/model.pt --evals hellaswag,piqa,lambada
    python evals/run_evals.py --checkpoint checkpoints/model.pt --evals glue --max-examples 100

Available evals:
    Intrinsic:    ppl, bpb, token_accuracy, positional_accuracy
    MC/Ranking:   hellaswag, piqa, arc_easy, arc_challenge, winogrande, boolq,
                  mmlu, truthfulqa, lambada, glue, superglue
    Generation:   xsum, squad, triviaqa, humaneval
    Probes:       niah, copy_retrieval
    Custom:       attention_analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import torch

# Workaround for PyTorch inductor bug
import torch._inductor.config as _inductor_config
_inductor_config.pattern_matcher = False

from evals.utils import load_model


# ── Eval registry ──────────────────────────────────────────────────────────

EVAL_REGISTRY = {}

def _register(name, module_path, fn_name, category):
    EVAL_REGISTRY[name] = {"module": module_path, "fn": fn_name, "category": category}

# Intrinsic
_register("ppl",                  "evals.intrinsic",           "eval_held_out_perplexity", "intrinsic")
_register("bpb",                  "evals.intrinsic",           "eval_bpb",                 "intrinsic")
_register("token_accuracy",      "evals.intrinsic",           "eval_token_accuracy",      "intrinsic")
_register("positional_accuracy", "evals.intrinsic",           "eval_positional_accuracy", "intrinsic")
_register("wikitext_ppl",        "evals.intrinsic",           "eval_wikitext_ppl",        "intrinsic")
_register("wikitext_bpb",        "evals.intrinsic",           "eval_wikitext_bpb",        "intrinsic")

# MC / Ranking benchmarks
_register("hellaswag",           "evals.benchmarks",          "eval_hellaswag",           "benchmark")
_register("piqa",                "evals.benchmarks",          "eval_piqa",                "benchmark")
_register("arc_easy",            "evals.benchmarks",          "eval_arc_easy",            "benchmark")
_register("arc_challenge",       "evals.benchmarks",          "eval_arc_challenge",       "benchmark")
_register("winogrande",          "evals.benchmarks",          "eval_winogrande",          "benchmark")
_register("boolq",               "evals.benchmarks",          "eval_boolq",               "benchmark")
_register("mmlu",                "evals.benchmarks",          "eval_mmlu",                "benchmark")
_register("truthfulqa",          "evals.benchmarks",          "eval_truthfulqa",          "benchmark")
_register("lambada",             "evals.benchmarks",          "eval_lambada",             "benchmark")
_register("lambada_enc_dec",     "evals.benchmarks",          "eval_lambada_enc_dec",     "benchmark")
_register("glue",                "evals.benchmarks",          "eval_glue",                "benchmark")
_register("superglue",           "evals.benchmarks",          "eval_superglue",           "benchmark")

# Generation
_register("xsum",                "evals.generation",          "eval_xsum",                "generation")
_register("squad",               "evals.generation",          "eval_squad",               "generation")
_register("triviaqa",            "evals.generation",          "eval_triviaqa",            "generation")
_register("humaneval",           "evals.generation",          "eval_humaneval",           "generation")

# Probes
_register("niah",                "evals.probes",              "eval_needle_in_haystack",  "probe")
_register("copy_retrieval",      "evals.probes",              "eval_copy_retrieval",      "probe")

# Custom
_register("attention_analysis",  "evals.attention_analysis",  "eval_attention_weights",   "custom")


# Preset groups
GROUPS = {
    "all":        list(EVAL_REGISTRY.keys()),
    "intrinsic":  [k for k, v in EVAL_REGISTRY.items() if v["category"] == "intrinsic"],
    "benchmarks": [k for k, v in EVAL_REGISTRY.items() if v["category"] == "benchmark"],
    "generation": [k for k, v in EVAL_REGISTRY.items() if v["category"] == "generation"],
    "probes":     [k for k, v in EVAL_REGISTRY.items() if v["category"] == "probe"],
    "quick":      ["ppl", "hellaswag", "piqa", "lambada", "token_accuracy"],
    "paper":      ["ppl", "bpb", "wikitext_ppl", "wikitext_bpb",
                   "hellaswag", "piqa", "arc_easy", "arc_challenge",
                   "winogrande", "lambada", "lambada_enc_dec", "boolq", "token_accuracy",
                   "positional_accuracy", "attention_analysis"],
}


def _load_eval_fn(name):
    """Lazy-import and return the eval function."""
    import importlib
    info = EVAL_REGISTRY[name]
    module = importlib.import_module(info["module"])
    return getattr(module, info["fn"])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluations on a trained block-based double-decoder model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer JSON (auto-detected if omitted)")
    parser.add_argument("--evals", default="quick",
                        help="Comma-separated eval names, or a group: all, intrinsic, benchmarks, "
                             "generation, probes, quick, paper")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap examples per eval (useful for fast debugging)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for GPU scoring (default 64, auto-halves on OOM)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (use if compile causes issues)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--eval-file", default="data/Pretrain/slimpajama_eval_packed.jsonl",
                        help="Held-out data file for intrinsic evals")
    args = parser.parse_args()

    # Resolve eval list
    eval_spec = args.evals.strip()
    if eval_spec in GROUPS:
        eval_names = GROUPS[eval_spec]
    else:
        eval_names = [e.strip() for e in eval_spec.split(",")]

    # Validate
    unknown = [e for e in eval_names if e not in EVAL_REGISTRY]
    if unknown:
        print(f"Unknown evals: {unknown}")
        print(f"Available: {sorted(EVAL_REGISTRY.keys())}")
        print(f"Groups: {sorted(GROUPS.keys())}")
        sys.exit(1)

    # Load model
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  Loading model from {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Evals: {eval_names}")
    print(f"{'='*60}\n")

    model, tokenizer, is_enc_dec = load_model(args.checkpoint, args.tokenizer, device)
    model_type = "Double_Decoder (enc-dec)" if is_enc_dec else "DecoderOnlyModel"
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {model_type}")
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    # torch.compile for faster forward passes
    if device.type == "cuda" and not args.no_compile:
        print(f"  Compiling model with torch.compile...")
        model = torch.compile(model, fullgraph=False, dynamic=False)
        print(f"  Compiled (first forward will be slower due to tracing)")
    print()

    # Run evals
    all_results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "parameters": param_count,
        "device": str(device),
        "evals": {},
    }

    for i, name in enumerate(eval_names):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(eval_names)}] Running: {name}")
        print(f"{'─'*60}")

        eval_fn = _load_eval_fn(name)
        start = time.time()

        try:
            # Build kwargs based on what the eval function accepts
            kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "is_enc_dec": is_enc_dec,
            }
            # Only pass kwargs the function actually accepts
            import inspect
            sig = inspect.signature(eval_fn)

            if args.max_examples is not None and "max_examples" in sig.parameters:
                kwargs["max_examples"] = args.max_examples
            if "num_examples" in sig.parameters and args.max_examples is not None:
                kwargs["num_examples"] = args.max_examples
            if "batch_size" in sig.parameters:
                kwargs["batch_size"] = args.batch_size
            if "eval_file" in sig.parameters:
                kwargs["eval_file"] = args.eval_file

            result = eval_fn(**kwargs)
            elapsed = time.time() - start
            result["time_seconds"] = round(elapsed, 1)
            all_results["evals"][name] = result

            # Print key metric
            _print_result(name, result, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results["evals"][name] = {"name": name, "error": str(e),
                                           "time_seconds": round(elapsed, 1)}

    # Summary
    print(f"\n{'='*60}")
    print("  EVAL SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results["evals"].items():
        if "error" in result:
            print(f"  {name:30s}  FAILED: {result['error'][:50]}")
        else:
            _print_result_short(name, result)
    print(f"{'='*60}\n")

    # Save
    output_path = args.output or f"evals/results_{os.path.basename(args.checkpoint).replace('.pt', '')}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}\n")


def _print_result(name, result, elapsed):
    """Print the primary metric for an eval result."""
    if "accuracy" in result:
        print(f"  ✓ {name}: {result['accuracy']:.4f} ({elapsed:.1f}s)")
    elif "perplexity" in result:
        print(f"  ✓ {name}: PPL={result['perplexity']:.2f} ({elapsed:.1f}s)")
    elif "bpb" in result:
        print(f"  ✓ {name}: BPB={result['bpb']:.4f} ({elapsed:.1f}s)")
    elif "f1" in result:
        print(f"  ✓ {name}: EM={result.get('exact_match', 0):.4f} F1={result['f1']:.4f} ({elapsed:.1f}s)")
    elif "pass_at_1" in result:
        print(f"  ✓ {name}: pass@1={result['pass_at_1']:.4f} ({elapsed:.1f}s)")
    elif "rouge1" in result:
        print(f"  ✓ {name}: R1={result['rouge1']:.4f} R2={result['rouge2']:.4f} RL={result['rougeL']:.4f} ({elapsed:.1f}s)")
    elif "average" in result:
        print(f"  ✓ {name}: avg={result['average']:.4f} ({elapsed:.1f}s)")
    elif "analysis" in result:
        print(f"  ✓ {name}: saved to {result.get('output_path', 'N/A')} ({elapsed:.1f}s)")
    else:
        print(f"  ✓ {name}: done ({elapsed:.1f}s)")


def _print_result_short(name, result):
    """One-line summary for final table."""
    if "accuracy" in result:
        print(f"  {name:30s}  acc={result['accuracy']:.4f}")
    elif "perplexity" in result:
        print(f"  {name:30s}  ppl={result['perplexity']:.2f}")
    elif "bpb" in result:
        print(f"  {name:30s}  bpb={result['bpb']:.4f}")
    elif "f1" in result:
        print(f"  {name:30s}  em={result.get('exact_match', 0):.4f}  f1={result['f1']:.4f}")
    elif "pass_at_1" in result:
        print(f"  {name:30s}  pass@1={result['pass_at_1']:.4f}")
    elif "rouge1" in result:
        print(f"  {name:30s}  R1={result['rouge1']:.4f}  R2={result['rouge2']:.4f}")
    elif "average" in result:
        print(f"  {name:30s}  avg={result['average']:.4f}")
    elif "overall_accuracy" in result:
        print(f"  {name:30s}  acc={result['overall_accuracy']:.4f}")
    elif "analysis" in result:
        for key, stats in result["analysis"].items():
            print(f"  {name:30s}  {key}: mean_dec_w={stats['overall_mean']:.4f}")
            break
    else:
        print(f"  {name:30s}  done")


if __name__ == "__main__":
    main()
