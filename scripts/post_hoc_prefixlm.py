#!/usr/bin/env python3
"""
The post_hoc_prefixlm.py script saves two artifacts per checkpoint to the HF model repo (default bpbradle/bbdd-scaling-checkpoints):

1. The post-prefixLM SFT'd weights (<basename>_postprefixlm_pct100.pt)

A torch-saved dict containing:
- state_dict — the SFT'd model weights
- arch — architecture config (dim, num_encoder_layers, num_decoder_layers)
- model_type — dd, sed, or dec
- vocab_size
- hparams — augmented from the original pretrain hparams with new fields:
  - phase: "post_hoc_prefixlm"
  - sft_objective: "prefixlm"
  - sft_tokens_target, sft_tokens_actual
  - sft_lr, sft_batch_size, sft_grad_accum
  - sft_train_file, sft_eval_file
  - source_pretrain_ckpt (filename of the input)

2. A JSON sidecar (<basename>_postprefixlm_pct100.json)

Plain JSON with the eval-comparison-ready stats:
- model_type, arch, arch_label, non_emb_params
- pretrain_ckpt (source filename)
- sft_objective: "prefixlm"
- Training: sft_train_loss, sft_train_time_s, sft_n_steps, sft_mfu_pct, sft_tokens_target, sft_tokens_actual, sft_lr, sft_batch_size, sft_grad_accum
- Eval (the headline number): eval_loss_held_out — token-weighted mean cross-entropy on the held-out prefixLM eval set, computed only over suffix positions (prefix labels are -100)
- eval_n_batches, eval_n_valid_tokens, eval_elapsed_s
- wallclock_s

Where on HF

Both files upload to: <hf-repo>/<model_type>/<arch_label>/<tok_label>tok/<filename>

For example: bpbradle/bbdd-scaling-checkpoints/dd/5M/6Btok/dd_5M_6Btok_postprefixlm_pct100.pt and the matching .json next to it.

This mirrors the path layout parallel_scaling.py uses for pretrain checkpoints, so post-prefixLM artifacts live in the same <mt>/<arch>/<tok>tok/ folder as their source .pt.

The JSON sidecar is the file you'd actually scan after a sweep — one per checkpoint, with eval_loss_held_out being the cross-architecture comparison number you're after.
"""

"""Post-hoc prefixLM SFT + held-out loss eval driver.

For every ``*_pct100.pt`` under ``--search-dir`` (skipping ``*_postprefixlm_*``),
this script sequentially:

  1. Loads the pretrain checkpoint and rebuilds the matching model.
  2. SFTs for ``--sft-tokens`` on packed SlimPajama prefixLM data using:
       - dd, sed: collators/encoder_decoder/prefixlm.py:PrefixLMCollator
       - dec:     collators/decoder/prefixlm.py:DecoderPrefixLMCollator
                  (loss-only-on-suffix; standard causal attention)
  3. Saves SFT'd weights as ``<basename>_postprefixlm_pct100.pt``.
  4. Computes mean cross-entropy loss on the held-out prefixLM eval set.
  5. Writes ``<basename>_postprefixlm_pct100.json`` sidecar with SFT loss
     curve + held-out eval loss.
  6. Optionally uploads weights + JSON sidecar to ``--hf-repo``.

The point: bring all three architectures (dd/sed/dec) onto a single training
objective (prefixLM) so their held-out loss numbers are comparable on one axis.
The pretrain phase trained them on different objectives (NTP, NTP, span
corruption); this script equalizes them post-hoc.

Inputs (produced by data/retrieval_scripts/slimpajama_prefixlm.py --pack):
    data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl    (~500M tokens)
    data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl   (~100M tokens)

Idempotent: --skip-existing skips checkpoints whose _postprefixlm sibling
already exists; safe to re-run after pod restarts. After each checkpoint
the SFT'd weights + JSON sidecar are uploaded to --hf-repo immediately —
if the pod dies mid-sweep, all completed work is durable.

Default behavior (no flags): discovers every *_pct100 in the canonical
HF model repo, downloads each, runs SFT + held-out eval, uploads results
back to the same repo at <model_type>/<arch>/<tok>tok/.

Examples:
    # Default — process every pretrain checkpoint on HF
    python scripts/post_hoc_prefixlm.py --skip-existing

    # Process only specific HF paths
    python scripts/post_hoc_prefixlm.py \\
        --checkpoints 'dd/5M/6Btok/pct100.pt,sed/5M/6Btok/pct100.pt'

    # Filter by fnmatch glob (applied to repo paths)
    python scripts/post_hoc_prefixlm.py --only-pattern '*5M*6B*'

    # Use local checkpoints instead of HF
    python scripts/post_hoc_prefixlm.py --local-only --search-dir checkpoints/
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from fnmatch import fnmatch
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the pretrain helpers so model construction, optimizer, and the
# train loop stay identical to parallel_scaling.py — any future change there
# is automatically inherited here.
from scripts.parallel_scaling import (
    build_model,
    sft_one_model,
    detect_gpu_tflops,
    _load_tuned_lrs,
    _load_tuned_wds,
    SEQ_LEN,
)
from training.hf_hub import upload_checkpoint, available as hf_available

from collators.encoder_decoder.prefixlm import PrefixLMCollator
from collators.decoder.prefixlm import DecoderPrefixLMCollator


# ── Wandb (optional, opt-in) ───────────────────────────────────────────────
# Lazy-imported so the script still runs in environments without wandb installed.
# All helpers no-op when --wandb-project is not set or wandb isn't importable.

_WANDB_OPTS = {"project": None, "entity": None, "prefix": None,
               "_module": None, "_imported": False}


def _wandb_module():
    if not _WANDB_OPTS["_imported"]:
        try:
            import wandb as _wb
            _WANDB_OPTS["_module"] = _wb
        except ImportError:
            print("[wandb] --wandb-project set but `wandb` not installed; "
                  "logging disabled. `pip install wandb` to enable.")
            _WANDB_OPTS["_module"] = None
        _WANDB_OPTS["_imported"] = True
    return _WANDB_OPTS["_module"]


def _wandb_enabled():
    return _WANDB_OPTS["project"] is not None and _wandb_module() is not None


def _wandb_init_run(model_type, arch_label, tok_label, config):
    """Open a new wandb run for one checkpoint's SFT pass. Returns the run
    object (or None if wandb is disabled / init failed)."""
    if not _wandb_enabled():
        return None
    wb = _wandb_module()
    prefix = _WANDB_OPTS["prefix"]
    name_parts = [p for p in (prefix, model_type, arch_label, f"{tok_label}tok") if p]
    run_name = "_".join(name_parts)
    try:
        return wb.init(
            project=_WANDB_OPTS["project"],
            entity=_WANDB_OPTS["entity"],
            name=run_name,
            group=f"{arch_label}_{tok_label}tok",
            tags=[model_type, arch_label, f"{tok_label}tok", "post_hoc_prefixlm"],
            config=config,
            reinit=True,
        )
    except Exception as e:
        print(f"[wandb] init failed for {run_name}: {e}")
        return None


def _wandb_finish_run(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        print(f"[wandb] run.finish() failed: {e}")


def _wandb_log(run, metrics, step=None):
    if run is None:
        return
    try:
        if step is not None:
            run.log(metrics, step=step)
        else:
            run.log(metrics)
    except Exception as e:
        print(f"[wandb] log failed: {e}")


# ── PrefixLM collator selector ─────────────────────────────────────────────

def build_prefixlm_collator(model_type, tokenizer, seq_len, seed):
    """Pick the right prefixLM collator for this model_type."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    common = dict(
        max_seq_len=seq_len,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        global_seed=seed,
    )
    if model_type == "dec":
        return DecoderPrefixLMCollator(**common)
    elif model_type in ("dd", "sed"):
        return PrefixLMCollator(**common)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Discovery + checkpoint loading (mirror post_hoc_sft.py) ────────────────

def find_pretrain_checkpoints(search_dir):
    """Return all *_pct100.pt under search_dir, excluding any postprefixlm/postsft
    siblings (those are derived files, not pretrain checkpoints).

    Matches both local naming `<basename>_pct100.pt` (e.g. dd_5M_6Btok_pct100.pt)
    and HF-cached naming `pct100.pt` (e.g. <local-dir>/dd/5M/6Btok/pct100.pt).
    """
    paths = []
    for pattern in ("*_pct100.pt", "pct100.pt"):
        paths.extend(Path(search_dir).rglob(pattern))
    paths = sorted(set(paths))
    return [p for p in paths
            if "_postsft_" not in p.name and "_postprefixlm_" not in p.name]


def postprefixlm_path_for(ckpt_path):
    """Map a *_pct100.pt path to its post-prefixLM sibling.

    Local naming (parallel_scaling.py output, has `_pct100` suffix):
        dd_5M_6Btok_pct100.pt → dd_5M_6Btok_postprefixlm_pct100.pt
    HF-cached naming (parallel_scaling.py uploads as `pct100.pt`):
        pct100.pt → postprefixlm_pct100.pt
    """
    name = ckpt_path.name
    if name.endswith("_pct100.pt"):
        new = name[: -len("_pct100.pt")] + "_postprefixlm_pct100.pt"
    elif name.endswith("pct100.pt"):
        new = name[: -len("pct100.pt")] + "postprefixlm_pct100.pt"
    else:
        # Fallback for non-standard names
        new = ckpt_path.stem + "_postprefixlm" + ckpt_path.suffix
    return ckpt_path.with_name(new)


def list_hf_repo_files(repo_id):
    """One-shot HfApi.model_info call. Returns the list of remote rfilenames.

    Pulled out so the caller can ask both 'what pretrain checkpoints exist?'
    and 'what post-SFT siblings already exist?' without two round-trips.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"[hf] listing files in {repo_id} (repo_type=model)...", flush=True)
    info = api.model_info(repo_id, files_metadata=True)
    siblings = getattr(info, "siblings", []) or []
    return [s.rfilename for s in siblings if getattr(s, "rfilename", None)]


def filter_pretrain_paths(rfilenames, only_pattern=None, explicit_paths=None):
    """From a list of remote rfilenames, return the pretrain *pct100.pt paths
    matching `only_pattern` (fnmatch glob) or `explicit_paths` (literal list).

    Excludes derived files. The marker test runs on the *basename* so it
    catches both naming conventions: local form `dd_5M_6Btok_postprefixlm_pct100.pt`
    (marker in the middle) and HF-cache form `<mt>/<arch>/<tok>tok/postprefixlm_pct100.pt`
    (marker at the start of the filename, with `/` not `_` in front). The
    earlier substring test `"_postprefixlm_" not in n` only caught the local
    form and silently let HF-form artifacts re-enter the pretrain list.
    """
    pretrain = [
        n for n in rfilenames
        if (n.endswith("_pct100.pt") or n.endswith("/pct100.pt") or n == "pct100.pt")
        and "postsft" not in n.rsplit("/", 1)[-1]
        and "postprefixlm" not in n.rsplit("/", 1)[-1]
    ]
    if explicit_paths is not None:
        wanted = set(explicit_paths)
        return sorted(p for p in pretrain if p in wanted)
    if only_pattern:
        return sorted(p for p in pretrain if fnmatch(p, only_pattern))
    return sorted(pretrain)


def existing_postprefixlm_set(rfilenames):
    """Set of remote rfilenames that look like post-SFT outputs from this script
    (matches both naming conventions: 'postprefixlm_pct100.pt' and
    '*_postprefixlm_pct100.pt'). Used by --skip-existing to check HF state."""
    return {
        n for n in rfilenames
        if (n.endswith("postprefixlm_pct100.pt")
            or n.endswith("_postprefixlm_pct100.pt"))
    }


def remote_postprefixlm_path_for(remote_pretrain_path):
    """Map a remote pretrain path → its expected post-SFT sibling path.
    Mirrors postprefixlm_path_for() but operates on string paths."""
    if remote_pretrain_path.endswith("_pct100.pt"):
        return remote_pretrain_path[:-len("_pct100.pt")] + "_postprefixlm_pct100.pt"
    if remote_pretrain_path.endswith("pct100.pt"):
        return remote_pretrain_path[:-len("pct100.pt")] + "postprefixlm_pct100.pt"
    return remote_pretrain_path  # fallback (shouldn't hit)


def download_one_checkpoint(repo_id, remote_path, local_dir):
    """Download a single rfilename from `repo_id` into `local_dir` preserving
    the repo's path structure. Returns the local Path. Idempotent — re-runs
    skip the network round-trip if the file already exists locally."""
    from huggingface_hub import hf_hub_download
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / remote_path
    if target.exists():
        print(f"  [hf-dl] cached: {remote_path}", flush=True)
    else:
        print(f"  [hf-dl] downloading: {remote_path}", flush=True)
        hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="model",
                        local_dir=str(local_dir))
    return target


def load_pretrain_checkpoint(ckpt_path, device):
    """Reconstruct the model from a parallel_scaling.py-format checkpoint.

    The checkpoint dict has no top-level 'arch' key — `arch` is reconstructed
    from `hparams['dim'/'num_encoder_layers'/'num_decoder_layers']`.
    Mirrors post_hoc_sft.py:load_pretrain_checkpoint.

    Returns (model, arch, model_type, hparams, raw_ckpt).
    """
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    hparams = raw.get("hparams", {})
    arch = {
        "dim": hparams["dim"],
        "num_encoder_layers": hparams["num_encoder_layers"],
        "num_decoder_layers": hparams["num_decoder_layers"],
    }
    model_type = raw["model_type"]
    vocab_size = raw["vocab_size"]
    # SED checkpoints store the *post-reduction* num_decoder_layers in hparams
    # (parallel_scaling.py overwrites arch["num_decoder_layers"] with
    # len(eager.decoder_layers) before saving). build_model's SED branch
    # re-applies max(1, (dec*3)//4), so passing the saved value verbatim
    # would shrink twice and produce too few layers to load the state_dict.
    #
    # Workaround without touching training: pass an inflated preimage to
    # build_model (smallest D such that (D*3)//4 == saved == ceil(saved*4/3)
    # = (saved*4 + 2)//3), then restore the truthful saved value on the arch
    # dict afterwards so downstream logging / wandb config see the actual
    # built layer count. NOTE: this preimage formula is load-bearingly tied
    # to the (dec*3)//4 reduction in parallel_scaling.build_model — update
    # both together if that formula ever changes.
    if model_type == "sed":
        saved_dec = arch["num_decoder_layers"]
        arch["num_decoder_layers"] = (saved_dec * 4 + 2) // 3
    # use_compile=True: ~30s warmup tax per checkpoint, but 2-3x steady-state
    # SFT throughput. Worth it now that --sft-tokens-frac scales budget with
    # pretrain (large cells run for tens of minutes — easy amortization).
    model = build_model(model_type, arch, vocab_size, device, use_compile=True)
    if model_type == "sed":
        arch["num_decoder_layers"] = saved_dec
    state_dict = raw["state_dict"]
    # Pretrain checkpoints save the EAGER state_dict (bare keys like
    # "embedding.weight"). After torch.compile, the wrapper renames every
    # parameter to "_orig_mod.embedding.weight". Load into the eager submodule
    # to bypass the prefix mismatch — load_state_dict on the compiled wrapper
    # would error with "missing _orig_mod.X" / "unexpected X" for every param.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    eager = getattr(model, "_orig_mod", model)
    eager.load_state_dict(state_dict)
    return model, arch, model_type, hparams, raw


def _parse_tok_label_to_int(label):
    """'6B' → 6_000_000_000, '500M' → 500_000_000, '125M' → 125_000_000.
    Returns None on failure (caller falls back to other sources)."""
    if not label or label == "?":
        return None
    label = str(label).strip()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = label[-1].upper()
    if suffix in multipliers:
        try:
            return int(float(label[:-1]) * multipliers[suffix])
        except (ValueError, IndexError):
            return None
    try:
        return int(label)
    except ValueError:
        return None


def get_pretrain_tokens(raw_ckpt, hparams, tok_label):
    """Best-available pretrain-tokens-seen count for a checkpoint.

    Priority:
      1. raw_ckpt['tokens_seen']  — actual tokens at save time, most accurate
      2. hparams['total_tokens']  — target budget for the pretrain run
      3. parse tok_label like '6B' → 6_000_000_000 (last-resort fallback)

    Raises ValueError if all three sources fail — caller should treat that
    as a non-recoverable config error rather than guessing.
    """
    if "tokens_seen" in raw_ckpt and raw_ckpt["tokens_seen"]:
        return int(raw_ckpt["tokens_seen"]), "raw_ckpt['tokens_seen']"
    if hparams.get("total_tokens"):
        return int(hparams["total_tokens"]), "hparams['total_tokens']"
    parsed = _parse_tok_label_to_int(tok_label)
    if parsed is not None:
        return parsed, f"parsed from tok_label={tok_label!r}"
    raise ValueError(
        f"Cannot determine pretrain tokens — none of "
        f"raw_ckpt['tokens_seen'], hparams['total_tokens'], or tok_label "
        f"({tok_label!r}) yielded a usable value")


def parse_run_id(ckpt_path, hparams):
    """Extract (model_type, arch_label, tok_label) from any available source.

    Priority: hparams (most reliable) → filename (`dd_5M_6Btok_pct100.pt`) →
    path (`<prefix>/dd/5M/6Btok/pct100.pt`, used by HF cache layout).

    Returns three strings; uses '?' for any field that can't be inferred.
    """
    mt = hparams.get("model_type")
    arch_label = hparams.get("arch_label")
    tok_label = hparams.get("tokens_label") or hparams.get("tok_label")

    # Try filename form: dd_5M_6Btok_pct100.pt
    if not (mt and arch_label and tok_label):
        parts = ckpt_path.stem.split("_")
        if len(parts) >= 4 and parts[2].endswith("tok"):
            mt = mt or parts[0]
            arch_label = arch_label or parts[1]
            tok_label = tok_label or parts[2][:-3]

    # Fall back to path form: .../<mt>/<arch>/<tok>tok/<filename>
    if not (mt and arch_label and tok_label):
        path_parts = ckpt_path.parts
        for i, seg in enumerate(path_parts):
            if seg.endswith("tok") and len(seg) > 3 and i >= 2:
                tok_label = tok_label or seg[:-3]
                arch_label = arch_label or path_parts[i - 1]
                mt = mt or path_parts[i - 2]
                break

    return mt or "?", arch_label or "?", tok_label or "?"


# ── Held-out eval: mean prefixLM loss on the eval dataloader ────────────────

@torch.no_grad()
def eval_prefixlm_loss(model, loader, device, max_batches=None):
    """Compute token-weighted mean cross-entropy loss on the held-out set.
    Returns (mean_loss, n_batches, n_valid_tokens, elapsed_s).

    Uses the same collator-supplied label masking as training, so loss is
    accumulated only over suffix positions (prefix labels are -100)."""
    eager = getattr(model, "_orig_mod", model)
    eager.eval()

    total_loss_weighted = 0.0
    total_n_valid = 0
    n_batches = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = eager(**batch)
            loss = out["loss"]
        n_valid = int((batch["labels"] != -100).sum().item())
        total_loss_weighted += float(loss.item()) * n_valid
        total_n_valid += n_valid
        n_batches += 1

    mean_loss = total_loss_weighted / max(total_n_valid, 1)
    return mean_loss, n_batches, total_n_valid, time.time() - t0


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("--search-dir", type=str, default="checkpoints/",
                   help="Root directory to recursively scan for *_pct100.pt")
    p.add_argument("--only-pattern", type=str, default=None,
                   help="fnmatch glob to filter checkpoint paths "
                        "(e.g. '*dd_50M*' or '*ben_sweep*')")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a ckpt if its _postprefixlm_pct100.pt sibling exists. "
                        "Safe to re-run after pod restarts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work and exit without loading models.")

    # PrefixLM SFT data + hyperparameters
    p.add_argument("--sft-train-file", type=str,
                   default="data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl",
                   help="Packed prefixLM training data (produced by "
                        "data/retrieval_scripts/slimpajama_prefixlm.py --pack)")
    p.add_argument("--sft-eval-file", type=str,
                   default="data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl",
                   help="Packed prefixLM held-out eval data")
    p.add_argument("--sft-tokens", type=int, default=None,
                   help="Explicit SFT token budget per checkpoint. Default None means "
                        "auto-derive as --sft-tokens-frac × this checkpoint's pretrain "
                        "tokens-seen — so a 100M-tok pretrain gets 10M SFT tokens, a "
                        "1B pretrain gets 100M, etc. Pass an int to override globally.")
    p.add_argument("--sft-tokens-frac", type=float, default=0.1,
                   help="Fraction of pretrain tokens-seen to use for SFT per checkpoint "
                        "(default 0.1 = 10%%). Ignored when --sft-tokens is set explicitly.")
    p.add_argument("--sft-lr", type=float, default=2e-4,
                   help="SFT base LR. μP-scales internally to "
                        "base × MUP_BASE_DIM/dim for hidden weights, so 2e-4 "
                        "→ ~2.5e-5 hidden LR at dim=512 (standard SFT range). "
                        "Previous default of 2e-5 left hidden LR at 2.5e-6, "
                        "which often left the SFT loss curve flat.")
    p.add_argument("--sft-grad-accum", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8,
                   help="SFT micro-batch size; auto-halved for dd/sed (which "
                        "carry both encoder and decoder activations through "
                        "backward, doubling per-batch memory headroom).")

    # Eval
    p.add_argument("--eval-batches", type=int, default=100,
                   help="Cap on eval batches per checkpoint (default 100 = "
                        "~1,600 sequences ≈ 3.3M tokens, plenty for stable mean "
                        "loss; SEM ~0.005 nats vs cross-arch differences ~0.05-0.5 "
                        "nats). Pass 0 to disable the cap and scan the full ~98M "
                        "eval tokens — useful for the publication run, but on "
                        "small SFT cells the full scan dominates wallclock "
                        "(~16 min vs ~2-3 sec capped).")
    p.add_argument("--eval-batch-size", type=int, default=16,
                   help="Eval batch size — can be larger than train since no backward")

    p.add_argument("--tokenizer-file", type=str,
                   default="tokenizer/tokenizer_32k.json")

    # HF source AND destination (defaults to the canonical checkpoints repo
    # so the script "just works" on a fresh pod with no flags).
    p.add_argument("--hf-repo", type=str,
                   default="bpbradle/bbdd-scaling-checkpoints",
                   help="HF model repo to (a) discover and download pretrain "
                        "checkpoints from and (b) upload the resulting "
                        "*_postprefixlm_pct100.pt + sidecar JSON back to. "
                        "Default matches the parallel_scaling.py upload target.")
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--checkpoints", type=str, default=None,
                   help="Comma-separated list of explicit paths-in-repo to "
                        "process (e.g. 'dd/5M/6Btok/pct100.pt,sed/5M/6Btok/pct100.pt'). "
                        "When set, skips repo-wide discovery and downloads only "
                        "these. Combine with the default --hf-repo for free.")
    p.add_argument("--local-only", action="store_true",
                   help="Skip HF discovery; scan --search-dir for local "
                        "*_pct100.pt files instead. Useful when checkpoints "
                        "are already on disk from a prior run.")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip the post-SFT upload step. Outputs stay on local "
                        "disk only — DON'T USE on an ephemeral pod unless you "
                        "have another persistence story.")
    p.add_argument("--keep-local", action="store_true",
                   help="Keep all .pt and .json files on local disk after each "
                        "checkpoint completes. Default behavior (in HF mode) is "
                        "to delete the pre-SFT .pt always (re-downloadable from "
                        "HF) and the post-SFT .pt + .json after a successful "
                        "upload — keeps peak local-disk usage to one pre + one "
                        "post checkpoint at a time, critical on small pods. "
                        "Pass --keep-local to retain everything for debugging "
                        "or for serving the post-SFT weights from local disk.")

    p.add_argument("--seed", type=int, default=42,
                   help="Global seed for deterministic SFT/eval data order.")

    # Wandb (optional, opt-in via --wandb-project)
    p.add_argument("--wandb-project", type=str, default=None,
                   help="If set, log each checkpoint's SFT run to wandb. One run "
                        "per checkpoint; per-step sft/loss + sft/lr + sft/mfu_pct "
                        "stream live, plus end-of-checkpoint eval/loss_held_out. "
                        "Skipped silently if wandb is not installed.")
    p.add_argument("--wandb-entity", type=str, default="block-based-double-decoders",
                   help="Wandb entity (team/user). Defaults to 'block-based-double-decoders' "
                        "so any --wandb-project name (including newly-created ones) lands "
                        "under that team. Override only if you're working in a different org.")
    p.add_argument("--wandb-run-name-prefix", type=str, default="prefixlm",
                   help="Prefix for per-checkpoint wandb run names "
                        "(<prefix>_<mt>_<arch_label>_<tok_label>tok). Default 'prefixlm'.")

    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Register wandb opts (no-op unless --wandb-project is set; lazy-imports
    # the module on first use so we don't pay the import cost on a no-wandb run).
    _WANDB_OPTS["project"] = args.wandb_project
    _WANDB_OPTS["entity"] = args.wandb_entity
    _WANDB_OPTS["prefix"] = args.wandb_run_name_prefix
    if args.wandb_project:
        if _wandb_module() is None:
            # Already printed an install hint inside _wandb_module(); proceed
            # with wandb disabled rather than aborting the whole sweep.
            _WANDB_OPTS["project"] = None
        else:
            print(f"[wandb] enabled  project={args.wandb_project}  "
                  f"entity={args.wandb_entity}  "
                  f"prefix={args.wandb_run_name_prefix}")

    # Read tuned LR/WD tables (same convention as parallel_scaling.py).
    _load_tuned_lrs()
    _load_tuned_wds()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_tflops = detect_gpu_tflops()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}  (peak BF16: {gpu_tflops:.0f} TFLOPS)")
    else:
        device = torch.device("cpu")
        gpu_tflops = 0.0
        print("WARN: no GPU detected; SFT will be unusably slow on CPU.")

    # ── HF preflight (fail fast on auth/typo before loading any models) ────
    upload_to_hf = bool(args.hf_repo) and not args.no_upload
    needs_hf = upload_to_hf or (not args.local_only)
    if needs_hf:
        if not hf_available():
            print("ERROR: HF needed but huggingface_hub not installed. "
                  "Run: pip install huggingface_hub")
            sys.exit(1)
        if upload_to_hf:
            from training.hf_hub import verify_repo
            try:
                verify_repo(args.hf_repo, private=args.hf_private, repo_type="model")
            except Exception as e:
                print(f"ERROR: HF preflight failed for {args.hf_repo}: {e}")
                sys.exit(1)

    # ── Discover work ──────────────────────────────────────────────────────
    # New default: list remote paths only — DO NOT download yet. We download
    # one checkpoint at a time inside the per-cell loop and (unless
    # --keep-local) delete it again after upload, so peak local-disk usage
    # stays at one pre-SFT + one post-SFT per cell.
    hf_local_dir = Path(args.search_dir) / "from_hf"
    # `ckpts_remote` is a list of either remote paths-in-repo (HF mode) or
    # local Paths (--local-only mode). The per-cell loop below downloads
    # the remote ones JIT and treats local ones as already-on-disk.
    ckpts_remote = []  # list of (remote_path: str) or (local_path: Path)
    is_hf_mode = False

    if args.local_only:
        ckpts_local_paths = find_pretrain_checkpoints(args.search_dir)
        if args.only_pattern:
            ckpts_local_paths = [p for p in ckpts_local_paths
                                 if fnmatch(str(p), args.only_pattern)]
        if args.skip_existing:
            ckpts_local_paths = [p for p in ckpts_local_paths
                                 if not postprefixlm_path_for(p).exists()]
        ckpts_remote = ckpts_local_paths  # already local; no download needed
    else:
        if not args.hf_repo:
            print("ERROR: --hf-repo required for HF discovery. "
                  "Pass --local-only to scan --search-dir instead.")
            sys.exit(1)
        is_hf_mode = True

        rfilenames = list_hf_repo_files(args.hf_repo)

        if args.checkpoints:
            requested = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
            if not requested:
                print("ERROR: --checkpoints was empty after splitting on commas")
                sys.exit(1)
            ckpts_remote = filter_pretrain_paths(rfilenames, explicit_paths=requested)
            missing = set(requested) - set(ckpts_remote)
            if missing:
                print(f"WARNING: {len(missing)} requested path(s) not found on HF: "
                      f"{sorted(missing)}")
        else:
            ckpts_remote = filter_pretrain_paths(rfilenames,
                                                 only_pattern=args.only_pattern)

        # --skip-existing now checks HF (the durable record) instead of local
        # disk (which is ephemeral in the new download-then-delete flow).
        if args.skip_existing:
            existing = existing_postprefixlm_set(rfilenames)
            before = len(ckpts_remote)
            ckpts_remote = [p for p in ckpts_remote
                            if remote_postprefixlm_path_for(p) not in existing]
            print(f"[skip-existing] {before - len(ckpts_remote)} cell(s) already "
                  f"have post-SFT weights on HF — skipping. {len(ckpts_remote)} "
                  f"left to process.")

        print(f"[hf] {len(ckpts_remote)} pretrain checkpoint(s) selected:",
              flush=True)
        for c in ckpts_remote:
            print(f"  - {c}")

    # For consistency below, we treat ckpts_remote uniformly:
    # in HF mode it's strings, in local mode it's Paths. The per-cell loop
    # branches on `is_hf_mode` to decide whether to download.
    ckpts = ckpts_remote

    if not ckpts:
        print(f"[plan] no checkpoints to process under {args.search_dir} "
              f"(pattern={args.only_pattern}, skip_existing={args.skip_existing})")
        return

    print(f"[plan] {len(ckpts)} checkpoint(s) to process:")
    for cp in ckpts:
        # cp is a remote string path (HF mode) or a local Path (--local-only).
        # Use the matching path-mapping helper so we don't AttributeError.
        if isinstance(cp, str):
            out_name = remote_postprefixlm_path_for(cp).rsplit("/", 1)[-1]
        else:
            out_name = postprefixlm_path_for(cp).name
        print(f"  - {cp}  →  {out_name}")

    if args.dry_run:
        print("[dry-run] exiting without running.")
        return

    # ── Tokenizer + datasets (load once, reuse for every checkpoint) ──────
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_file)
    print(f"[data] loading SFT data from {args.sft_train_file}...")
    sft_train_ds = load_dataset("json", data_files=args.sft_train_file, split="train")
    print(f"[data] loaded {len(sft_train_ds):,} packed SFT sequences "
          f"(~{len(sft_train_ds) * SEQ_LEN / 1e6:.0f}M tokens)")
    print(f"[data] loading eval data from {args.sft_eval_file}...")
    eval_ds = load_dataset("json", data_files=args.sft_eval_file, split="train")
    print(f"[data] loaded {len(eval_ds):,} packed eval sequences "
          f"(~{len(eval_ds) * SEQ_LEN / 1e6:.0f}M tokens)")

    sft_train_ds = sft_train_ds.shuffle(seed=args.seed)
    # Eval not shuffled — deterministic ordering for reproducible loss numbers.

    # ── Per-checkpoint loop (sequential, OOM-safe via finally) ─────────────
    n_ok = 0
    n_fail = 0
    summary_rows = []
    for i, ckpt_remote_or_local in enumerate(ckpts):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ckpts)}] {ckpt_remote_or_local}")
        print(f"{'='*70}")
        t_total = time.time()
        model = None  # set inside try, freed in finally
        wandb_run = None  # set inside try, finished in finally
        ckpt_path = None  # local Path to the pre-SFT .pt; set after download
        postpfx_path = None  # set once we know the input filename
        sidecar_path = None  # ditto
        upload_succeeded = False

        try:
            # JIT-download in HF mode so peak local-disk usage stays small.
            # In --local-only mode the path is already a local Path.
            if is_hf_mode:
                ckpt_path = download_one_checkpoint(
                    args.hf_repo, ckpt_remote_or_local, hf_local_dir)
            else:
                ckpt_path = ckpt_remote_or_local

            print(f"  [load] reading {ckpt_path}...")
            model, arch, model_type, hparams, raw_ckpt = load_pretrain_checkpoint(
                ckpt_path, device)
            ne = (sum(p.numel() for p in model.parameters())
                  - model.embedding.weight.numel())
            # parse_run_id pulls from hparams first, then filename, then path
            # — handles both local (`dd_5M_6Btok_pct100.pt`) and HF cache layout
            # (`<prefix>/dd/5M/6Btok/pct100.pt`).
            _mt_inf, _arch_inf, tok_label = parse_run_id(ckpt_path, hparams)
            arch_label = hparams.get("arch_label") or _arch_inf
            display = f"{model_type}_{arch_label or 'unknown'}"
            print(f"  [load] {display}  dim={arch['dim']}  "
                  f"enc/dec={arch['num_encoder_layers']}/{arch['num_decoder_layers']}  "
                  f"non_emb={ne:,}  vocab={raw_ckpt['vocab_size']:,}  "
                  f"pretrain_tok={tok_label}")

            # ── Determine SFT token budget for this checkpoint ─────────────
            # Default behavior: use --sft-tokens-frac × pretrain tokens-seen.
            # --sft-tokens overrides if set (same value for every checkpoint).
            if args.sft_tokens is not None:
                sft_tokens = args.sft_tokens
                sft_tokens_source = f"--sft-tokens override ({sft_tokens:,})"
            else:
                pretrain_tokens, src = get_pretrain_tokens(raw_ckpt, hparams, tok_label)
                sft_tokens = int(pretrain_tokens * args.sft_tokens_frac)
                sft_tokens_source = (f"{args.sft_tokens_frac*100:.0f}% × {pretrain_tokens:,} "
                                     f"pretrain tokens (from {src})")

            # ── PrefixLM SFT setup ─────────────────────────────────────────
            # Halve bs / double ga for enc-dec to keep effective batch the
            # same while staying inside memory headroom (mirrors
            # parallel_scaling.py:1169-1174 and post_hoc_sft.py:418-423).
            if model_type in ("dd", "sed"):
                sft_bs = max(1, args.batch_size // 2)
                sft_ga = max(1, args.sft_grad_accum * 2)
            else:
                sft_bs = args.batch_size
                sft_ga = args.sft_grad_accum

            sft_collator = build_prefixlm_collator(
                model_type, tokenizer, SEQ_LEN, seed=args.seed)
            sft_loader = DataLoader(
                sft_train_ds, batch_size=sft_bs, shuffle=True,
                collate_fn=sft_collator, drop_last=True,
                num_workers=2, pin_memory=True,
            )
            sft_total_micro = max(sft_ga,
                                  sft_tokens // (sft_bs * SEQ_LEN))
            # Natural cap: don't request more batches than the loader can
            # provide. With drop_last=True this is len(sft_train_ds)//sft_bs.
            # Protects against pretrain_tokens × sft_tokens_frac > data file size.
            sft_total_micro = min(sft_total_micro, len(sft_loader))
            sft_steps = sft_total_micro // sft_ga

            print(f"  [sft] {display}: ~{sft_tokens/1e6:.0f}M tokens  "
                  f"[budget: {sft_tokens_source}]  "
                  f"({sft_total_micro} micro / {sft_steps} steps  "
                  f"bs={sft_bs} ga={sft_ga} lr={args.sft_lr})")

            # Init a wandb run for this checkpoint. Per-step metrics will
            # stream via the on_step callback below; eval + summary metrics
            # are logged after eval completes.
            wandb_run = _wandb_init_run(
                model_type=model_type,
                arch_label=arch_label or "unknown",
                tok_label=tok_label,
                config={
                    "model_type": model_type,
                    "arch_label": arch_label,
                    "tok_label": tok_label,
                    "non_emb_params": ne,
                    "dim": arch["dim"],
                    "num_encoder_layers": arch["num_encoder_layers"],
                    "num_decoder_layers": arch["num_decoder_layers"],
                    "vocab_size": raw_ckpt["vocab_size"],
                    "sft_tokens_target": sft_tokens,
                    "sft_lr": args.sft_lr,
                    "sft_batch_size": sft_bs,
                    "sft_grad_accum": sft_ga,
                    "sft_objective": "prefixlm",
                    "source_pretrain_ckpt": ckpt_path.name,
                    "sft_train_file": args.sft_train_file,
                    "sft_eval_file": args.sft_eval_file,
                    "phase": "post_hoc_prefixlm",
                },
            )

            # Per-step callback — receives (step, dict) from sft_one_model
            # at every log_interval. Closes over wandb_run so it can be None
            # without breaking the SFT loop.
            def _wandb_step_cb(step, metrics):
                _wandb_log(wandb_run, metrics, step=step)

            sft_t0 = time.time()
            avg_sft_loss, sft_train_time, sft_n_steps, sft_mfu = sft_one_model(
                {"model": model, "arch": arch, "model_type": model_type,
                 "ne": ne, "display": display},
                sft_loader, device, sft_total_micro, sft_ga, args.sft_lr,
                gpu_tflops=gpu_tflops, tokens_per_micro=sft_bs * SEQ_LEN,
                on_step=_wandb_step_cb,
            )
            print(f"  [sft] done: train_loss={avg_sft_loss:.3f}  "
                  f"time={sft_train_time:.0f}s  MFU={sft_mfu:.1f}%")

            # ── Held-out eval (same collator, fresh DataLoader, no shuffle) ─
            # Eval batch can be larger since no backward — use args.eval_batch_size.
            # For enc-dec models we still halve to keep memory parity; eval
            # peaks lower than train but the safety margin matters here too.
            eval_bs = max(1, args.eval_batch_size // 2) if model_type in ("dd", "sed") \
                      else args.eval_batch_size
            eval_collator = build_prefixlm_collator(
                model_type, tokenizer, SEQ_LEN, seed=args.seed + 1)
            eval_loader = DataLoader(
                eval_ds, batch_size=eval_bs, shuffle=False,
                collate_fn=eval_collator, drop_last=True,
                num_workers=2, pin_memory=True,
            )
            # --eval-batches 0 means "scan everything"; pass None to the eval
            # loop in that case so it doesn't break out immediately.
            eval_cap = args.eval_batches if args.eval_batches and args.eval_batches > 0 else None
            print(f"  [eval] held-out prefixLM loss (bs={eval_bs}"
                  + (f", capped at {eval_cap} batches" if eval_cap else ", full scan")
                  + ")...")
            eval_loss, eval_n_batches, eval_n_valid, eval_elapsed = eval_prefixlm_loss(
                model, eval_loader, device, max_batches=eval_cap)
            print(f"  [eval] held-out loss = {eval_loss:.4f}  "
                  f"({eval_n_batches} batches, {eval_n_valid:,} suffix tokens, "
                  f"{eval_elapsed:.0f}s)")

            # Log end-of-checkpoint summary metrics to wandb. Per-step sft/* came
            # from the on_step callback above; this is the headline eval result.
            _wandb_log(wandb_run, {
                "eval/loss_held_out": float(eval_loss),
                "eval/n_batches": int(eval_n_batches),
                "eval/n_valid_tokens": int(eval_n_valid),
                "eval/elapsed_s": float(eval_elapsed),
                "sft/final_loss": float(avg_sft_loss),
                "sft/total_train_time_s": float(sft_train_time),
                "sft/n_optimizer_steps": int(sft_n_steps),
                "sft/final_mfu_pct": float(sft_mfu),
            })

            # ── Save SFT'd weights + JSON sidecar ──────────────────────────
            postpfx_path = postprefixlm_path_for(ckpt_path)
            sft_hparams = dict(hparams)
            sft_hparams.update({
                "phase": "post_hoc_prefixlm",
                "sft_objective": "prefixlm",
                "sft_tokens_target": sft_tokens,
                "sft_tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                "sft_lr": args.sft_lr,
                "sft_batch_size": sft_bs,
                "sft_grad_accum": sft_ga,
                "sft_train_file": args.sft_train_file,
                "sft_eval_file": args.sft_eval_file,
                "source_pretrain_ckpt": ckpt_path.name,
            })
            eager = getattr(model, "_orig_mod", model)
            torch.save({
                "state_dict": eager.state_dict(),
                "arch": arch,
                "model_type": model_type,
                "hparams": sft_hparams,
                "vocab_size": raw_ckpt["vocab_size"],
            }, str(postpfx_path))
            print(f"  [save] weights → {postpfx_path}")

            sidecar = {
                "model_type": model_type,
                "arch": arch,
                "arch_label": arch_label,
                "non_emb_params": ne,
                "pretrain_ckpt": ckpt_path.name,
                "sft_objective": "prefixlm",
                "sft_train_loss": avg_sft_loss,
                "sft_train_time_s": sft_train_time,
                "sft_n_steps": sft_n_steps,
                "sft_mfu_pct": sft_mfu,
                "sft_tokens_target": sft_tokens,
                "sft_tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                "sft_lr": args.sft_lr,
                "sft_batch_size": sft_bs,
                "sft_grad_accum": sft_ga,
                "eval_loss_held_out": eval_loss,
                "eval_n_batches": eval_n_batches,
                "eval_n_valid_tokens": eval_n_valid,
                "eval_elapsed_s": eval_elapsed,
                "wallclock_s": time.time() - t_total,
            }
            sidecar_path = postpfx_path.with_suffix(".json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2))
            print(f"  [save] sidecar → {sidecar_path}")

            # ── HF upload ──────────────────────────────────────────────────
            if upload_to_hf:
                n_uploaded = 0
                for f in (postpfx_path, sidecar_path):
                    try:
                        # training/hf_hub.upload_checkpoint signature:
                        #   (local_path, repo_id, path_in_repo, repo_type="model", ...)
                        # Note: no `private` kwarg — repo creation/visibility is
                        # handled by verify_repo() at startup, not per-file upload.
                        upload_checkpoint(
                            local_path=f,
                            repo_id=args.hf_repo,
                            path_in_repo=f"{model_type}/{arch_label or 'unknown'}/"
                                         f"{tok_label}tok/{f.name}",
                            repo_type="model",
                        )
                        print(f"  [hf-upload] ✓ {f.name}")
                        n_uploaded += 1
                    except Exception as e:
                        print(f"  [hf-upload] FAILED for {f}: {e}  "
                              "(weights still on local disk; manual retry possible)")
                # Both files (post-SFT .pt + .json sidecar) must upload for us
                # to safely delete the locals. If either fails, keep them.
                upload_succeeded = (n_uploaded == 2)
            else:
                # --no-upload: by definition we want to keep the locals
                upload_succeeded = False

            n_ok += 1
            summary_rows.append({
                "ckpt": ckpt_path.name,
                "model_type": model_type,
                "arch_label": arch_label,
                "sft_train_loss": avg_sft_loss,
                "eval_loss_held_out": eval_loss,
                "wallclock_s": time.time() - t_total,
            })
            print(f"  [done] total wallclock {time.time() - t_total:.0f}s")

        except Exception as e:
            print(f"  [ERROR] failed processing {ckpt_path}: {e}")
            traceback.print_exc()
            print(f"  [ERROR] continuing to next checkpoint")
            n_fail += 1
        finally:
            # Free GPU memory before the next checkpoint loads. Without this
            # the previous model's params + optimizer state stay resident
            # until Python gc runs, which on a 24 GB card is the difference
            # between fitting the next 100M model and OOMing.
            if model is not None:
                del model
            torch.cuda.empty_cache()
            # Close the wandb run for this checkpoint so its history flushes
            # before we move on. Even on exception, finishing prevents a
            # zombie run from blocking the next checkpoint's init.
            _wandb_finish_run(wandb_run)

            # Disk cleanup. Only active in HF mode (where files were just
            # downloaded and are re-fetchable) and only when --keep-local
            # is NOT set. Skipped entirely in --local-only mode so the user's
            # pre-existing checkpoints aren't accidentally deleted.
            if is_hf_mode and not args.keep_local:
                # Pre-SFT .pt: always safe to delete after the cell — it's
                # already on HF (that's where we just downloaded it from).
                if ckpt_path is not None and ckpt_path.exists():
                    try:
                        ckpt_path.unlink()
                        print(f"  [cleanup] deleted pre-SFT: {ckpt_path.name} "
                              f"({ckpt_path.parent})", flush=True)
                    except Exception as e:
                        print(f"  [cleanup] failed to delete {ckpt_path}: {e}")
                # Post-SFT .pt + .json: delete only if both uploaded
                # successfully. Otherwise keep them so the user can retry
                # the upload manually instead of redoing SFT.
                if upload_succeeded:
                    for f in (postpfx_path, sidecar_path):
                        if f is None or not f.exists():
                            continue
                        try:
                            f.unlink()
                            print(f"  [cleanup] deleted post-SFT: {f.name}",
                                  flush=True)
                        except Exception as e:
                            print(f"  [cleanup] failed to delete {f}: {e}")
                elif (postpfx_path is not None and postpfx_path.exists()):
                    print(f"  [cleanup] KEEPING post-SFT (upload incomplete): "
                          f"{postpfx_path.name}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[summary] {n_ok} succeeded, {n_fail} failed, "
          f"{len(ckpts) - n_ok - n_fail} skipped")
    print(f"{'='*70}")
    if summary_rows:
        print(f"\n  {'model_type':<12} {'arch':<10} {'sft_loss':>10} "
              f"{'eval_loss':>10} {'wallclock':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in summary_rows:
            print(f"  {r['model_type']:<12} {str(r['arch_label'] or '?'):<10} "
                  f"{r['sft_train_loss']:>10.3f} {r['eval_loss_held_out']:>10.4f} "
                  f"{r['wallclock_s']:>9.0f}s")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
