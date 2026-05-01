"""Interactive REPL for a trained DecoderOnlyModel checkpoint.

This is a *base pretrain* model — it CONTINUES text rather than chatting.
Type a prompt; the model emits a continuation. Prior turns stay in context
so the model can build on what it generated. Commands: 'reset', 'exit'.

Usage:
    # 1) Re-run training with weights persisted (one-line patch added):
    python scripts/parallel_scaling.py --arch-set large --only-arch 5M \\
        --token-set large --token-budgets 500M --model-types dec \\
        --auto-batch-size --no-mup --peak-lr 1e-3 --save-checkpoints \\
        --wandb-project scaling-laws-runpod

    # 2) Chat with the resulting checkpoint:
    python scripts/chat.py --ckpt checkpoints/parallel_scaling/dec_5M_500Mtok.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# Allow `python scripts/chat.py` from the repo root without setting PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder import DecoderOnlyModel  # noqa: E402


def load_model(ckpt_path: Path, device: torch.device):
    """Reconstruct DecoderOnlyModel from a parallel_scaling.py or pretrain.py
    checkpoint. Both formats stash a `state_dict` and an `hparams` dict; this
    handles either layer-naming convention (num_layers vs enc/dec split) and
    strips torch.compile / DDP key prefixes."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt["hparams"]
    vocab_size = ckpt.get("vocab_size") or state_dict["embedding.weight"].shape[0]

    num_layers = hparams.get("num_layers")
    if num_layers is None:
        num_layers = (hparams.get("num_encoder_layers", 0)
                      + hparams.get("num_decoder_layers", 0))
    if num_layers == 0:
        raise ValueError(f"Couldn't infer num_layers from hparams: {hparams}")

    model = DecoderOnlyModel(
        vocab_size=vocab_size,
        dim=hparams["dim"],
        num_heads=hparams.get("num_heads", hparams["dim"] // 64),
        num_layers=num_layers,
        seq_len=hparams.get("seq_len", 2048),
        mup_base_dim=hparams.get("mup_base_dim", 0),
    )

    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.to(device).eval()
    return model, hparams, vocab_size


@torch.inference_mode()
def stream_generate(model, tokens, max_new, temperature, top_k, top_p,
                    eos_id, seq_len, device):
    """Yield token ids one at a time. `tokens` is a CPU 1D LongTensor of context.
    Truncates context from the left to seq_len — fine because attention is
    causal, so dropping old tokens just shortens the window."""
    for _ in range(max_new):
        ctx = tokens[-seq_len:].unsqueeze(0).to(device)
        logits = model(input_ids=ctx)["logits"][0, -1].float()
        logits = logits / max(temperature, 1e-5)

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")

        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cum > top_p
            # Always keep at least the top-1 token
            cutoff[..., 0] = False
            sorted_logits[cutoff] = -float("inf")
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        tokens = torch.cat([tokens, torch.tensor([next_id], dtype=tokens.dtype)])
        yield next_id
        if next_id == eos_id:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="Trained .pt (e.g. checkpoints/parallel_scaling/dec_5M_500Mtok.pt)")
    ap.add_argument("--tokenizer", default=PROJECT_ROOT / "tokenizer" / "tokenizer.json",
                    type=Path)
    ap.add_argument("--max-new", type=int, default=128,
                    help="Max new tokens to generate per turn.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-context", type=int, default=None,
                    help="Cap prompt+history tokens (default: model's seq_len).")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Loading {args.ckpt} on {device}…")
    model, hparams, vocab_size = load_model(args.ckpt, device)
    seq_len = args.max_context or hparams.get("seq_len", 2048)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    if tokenizer.vocab_size != vocab_size:
        print(f"[warn] tokenizer vocab={tokenizer.vocab_size} but model "
              f"vocab={vocab_size} — make sure --tokenizer matches the run.")

    n_layers = (hparams.get("num_layers")
                or hparams.get("num_encoder_layers", 0)
                + hparams.get("num_decoder_layers", 0))
    print("=" * 64)
    print(f"  Model:  {hparams.get('model_cls', 'DecoderOnly')}  "
          f"dim={hparams['dim']}  layers={n_layers}  seq_len={seq_len}  "
          f"vocab={vocab_size}")
    print(f"  Sample: T={args.temperature}  top_k={args.top_k}  top_p={args.top_p}  "
          f"max_new={args.max_new}")
    print("  This is a base pretrain model — it continues text, not chats.")
    print("  Commands:  reset (clear history)   exit (quit)")
    print("=" * 64)

    history = torch.tensor([bos_id], dtype=torch.long)

    while True:
        try:
            user = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "exit":
            break
        if user == "reset":
            history = torch.tensor([bos_id], dtype=torch.long)
            print("[history cleared]")
            continue

        # ByteLevel BPE: prepend a space so the first content token gets its
        # leading-space marker, matching how packed pretrain text is tokenized.
        new_ids = tokenizer.encode(" " + user, add_special_tokens=False)
        history = torch.cat([history, torch.tensor(new_ids, dtype=torch.long)])

        # Stream: re-decode the full output buffer each step and print only
        # the newly-rendered suffix. Imperfect with multi-byte BPE pieces but
        # good enough for English text and avoids buffering the whole turn.
        out_ids: list[int] = []
        printed = 0
        print("    ", end="", flush=True)
        for nid in stream_generate(
            model, history, args.max_new, args.temperature, args.top_k, args.top_p,
            eos_id, seq_len, device,
        ):
            out_ids.append(nid)
            full = tokenizer.decode(out_ids, skip_special_tokens=False)
            chunk = full[printed:]
            if chunk:
                print(chunk, end="", flush=True)
                printed = len(full)
        print()

        history = torch.cat([history, torch.tensor(out_ids, dtype=torch.long)])
        # If the model emitted </s>, start a fresh document so the next turn
        # gets a clean BOS-anchored context.
        if out_ids and out_ids[-1] == eos_id:
            history = torch.cat([history, torch.tensor([bos_id], dtype=torch.long)])


if __name__ == "__main__":
    main()
