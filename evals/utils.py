"""
Core utilities for evaluating block-based double-decoder models.
Handles model loading, log-likelihood scoring, and text generation
for both Double_Decoder (encoder-decoder) and DecoderOnlyModel architectures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


def load_model(checkpoint_path, tokenizer_path=None, device="cuda"):
    """Load a trained model and tokenizer from a checkpoint.

    Returns: (model, tokenizer, is_enc_dec)
    """
    from models.double_decoder import Double_Decoder
    from models.decoder import DecoderOnlyModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt["hparams"]

    # Handle torch.compile wrapper keys
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Determine model type from hparams
    is_enc_dec = "num_encoder_layers" in hparams and "num_decoder_layers" in hparams

    if is_enc_dec:
        model = Double_Decoder(**hparams)
    else:
        model = DecoderOnlyModel(**hparams)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    if tokenizer_path is None:
        for candidate in ["tokenizer/tokenizer_32k.json", "tokenizer/tokenizer.json"]:
            if os.path.exists(candidate):
                tokenizer_path = candidate
                break
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    return model, tokenizer, is_enc_dec


# ── Log-likelihood scoring ─────────────────────────────────────────────────

@torch.no_grad()
def get_log_probs(model, tokenizer, context_ids, continuation_ids, device, is_enc_dec):
    """Per-token log probs of continuation given context.

    Returns: tensor of shape [len(continuation_ids)]
    """
    if is_enc_dec:
        return _log_probs_enc_dec(model, tokenizer, context_ids, continuation_ids, device)
    else:
        return _log_probs_decoder_only(model, tokenizer, context_ids, continuation_ids, device)


def _log_probs_decoder_only(model, tokenizer, context_ids, continuation_ids, device):
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    input_ids = [bos_id] + context_ids + continuation_ids
    input_t = torch.tensor([input_ids], device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=input_t)["logits"]

    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    # logits[t] predicts input[t+1]; continuation starts at index 1+len(context)
    cont_start = 1 + len(context_ids)
    return torch.stack([
        log_probs[cont_start - 1 + i, continuation_ids[i]]
        for i in range(len(continuation_ids))
    ])


def _log_probs_enc_dec(model, tokenizer, context_ids, continuation_ids, device):
    bos_id = tokenizer.convert_tokens_to_ids("<s>")

    enc_ids = [bos_id] + context_ids
    dec_ids = [bos_id] + continuation_ids  # BOS as decoder start token

    encoder_input_ids = torch.tensor([enc_ids], device=device)
    decoder_input_ids = torch.tensor([dec_ids], device=device)

    enc_len = len(enc_ids)
    dec_len = len(dec_ids)

    decoder_input_positions = torch.arange(enc_len, enc_len + dec_len, device=device).unsqueeze(0)
    seq_len = getattr(model, "seq_len", 2048)
    decoder_input_positions = decoder_input_positions.clamp(max=seq_len - 1)

    blocks = torch.tensor([enc_len], device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_input_positions=decoder_input_positions,
            blocks=blocks,
            sft=True,
        )["logits"]

    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    # logits[i] predicts continuation[i] (since decoder = [BOS, c0, c1, ...])
    return torch.stack([
        log_probs[i, continuation_ids[i]]
        for i in range(len(continuation_ids))
    ])


# ── Multiple-choice scoring ───────────────────────────────────────────────

def score_choices(model, tokenizer, context_str, choices, device, is_enc_dec):
    """Score continuation choices. Returns (best_idx, list_of_scores).

    Scores are length-normalized mean log-probs.
    """
    context_ids = tokenizer.encode(context_str, add_special_tokens=False)
    scores = []
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        if not choice_ids:
            scores.append(float("-inf"))
            continue
        lp = get_log_probs(model, tokenizer, context_ids, choice_ids, device, is_enc_dec)
        scores.append(lp.sum().item() / len(choice_ids))

    best = max(range(len(scores)), key=lambda i: scores[i])
    return best, scores


# ── Full-sequence log probs (for PPL / BPB) ───────────────────────────────

@torch.no_grad()
def get_sequence_log_probs(model, tokenizer, input_ids_list, device, is_enc_dec):
    """Per-token log probs for a packed sequence (pretraining-style forward).

    Returns: (log_probs tensor [seq_len-1], predicted_ids tensor [seq_len-1])
    """
    input_t = torch.tensor([input_ids_list], device=device)

    if is_enc_dec:
        seq_len = len(input_ids_list)
        blocks = torch.tensor([seq_len // 2], device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t, blocks=blocks, sft=False)["logits"]
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t)["logits"]

    logits = logits[0].float()  # [seq_len, vocab]
    log_probs = F.log_softmax(logits, dim=-1)

    targets = torch.tensor(input_ids_list[1:], device=device)
    token_lps = log_probs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    predicted = logits[:-1].argmax(dim=-1)

    return token_lps, predicted


# ── Text generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens, device, is_enc_dec,
                  temperature=0.0, top_k=50):
    """Generate text from the model. temperature=0 for greedy."""
    if is_enc_dec:
        return _gen_enc_dec(model, tokenizer, prompt, max_new_tokens, device, temperature, top_k)
    else:
        return _gen_decoder_only(model, tokenizer, prompt, max_new_tokens, device, temperature, top_k)


def _sample_next(logits, temperature, top_k):
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < vals[:, -1:]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def _gen_decoder_only(model, tokenizer, prompt, max_new_tokens, device, temperature, top_k):
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")

    ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_t = torch.tensor([ids], device=device)
    prompt_len = len(ids)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t)["logits"][:, -1:, :]
        next_tok = _sample_next(logits.squeeze(1), temperature, top_k)
        input_t = torch.cat([input_t, next_tok], dim=-1)
        if next_tok.item() == eos_id:
            break

    gen_ids = input_t[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _gen_enc_dec(model, tokenizer, prompt, max_new_tokens, device, temperature, top_k):
    from components.block_masks import create_inference_masks

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    assistant_id = tokenizer.convert_tokens_to_ids("<assistant>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")

    enc_ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    enc_input = torch.tensor([enc_ids], device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        enc_out = model.encode(enc_input)

    dec_ids = torch.tensor([[assistant_id]], device=device)
    enc_len = len(enc_ids)
    dec_pos = torch.tensor([[enc_len]], device=device)

    for _ in range(max_new_tokens):
        num_dec = dec_ids.shape[1]
        try:
            masks = create_inference_masks(device=device, enc_len=enc_len, dec_len=num_dec)
        except Exception:
            masks = None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.decode(dec_ids, enc_out, masks, dec_pos)

        next_tok = _sample_next(logits[:, -1, :], temperature, top_k)
        dec_ids = torch.cat([dec_ids, next_tok], dim=-1)
        dec_pos = torch.cat([dec_pos, dec_pos[:, -1:] + 1], dim=1)

        if next_tok.item() == eos_id:
            break

    gen_ids = dec_ids[0, 1:].tolist()  # skip assistant token
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
