"""
Core utilities for evaluating block-based double-decoder models.
Handles model loading, log-likelihood scoring (single + batched),
and text generation for both Double_Decoder and DecoderOnlyModel.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


def _sample_pretrain_blocks(max_len, device, seed=42):
    """Sample block boundaries matching the pretrain collator distribution.

    Training (BasicPretrainCollator) uses:
        block_num = rng.integers(2, 8)          # 2-7 boundaries → 3-8 blocks
        blocks = sorted sample from [2, seq_len-2]

    Using a fixed seed ensures eval is deterministic across runs.
    """
    rng = np.random.default_rng(seed)
    block_num = int(rng.integers(2, 8))  # 2-7 boundaries, matching training
    # Ensure enough positions exist for the requested boundaries
    available = max(max_len - 3, 0)  # usable positions in [2, max_len-2]
    block_num = min(block_num, available)
    if block_num < 1:
        # Sequence too short for proper blocks — fall back to midpoint
        return torch.tensor([max(1, max_len // 2)], dtype=torch.int64, device=device)
    boundaries = np.sort(
        rng.choice(np.arange(2, max_len - 1), size=block_num, replace=False)
    )
    return torch.tensor(boundaries, dtype=torch.int64, device=device)


def load_model(checkpoint_path, tokenizer_path=None, device="cuda"):
    """Load a trained model and tokenizer from a checkpoint.

    Returns: (model, tokenizer, is_enc_dec)
    """
    from models.double_decoder import Double_Decoder
    from models.decoder import DecoderOnlyModel
    from models.standard_enc_dec import StandardEncDec

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt["hparams"]

    # Handle torch.compile wrapper keys
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Determine model type from state_dict keys (hparams can be ambiguous
    # because DecoderOnlyModel checkpoints still contain num_encoder_layers)
    has_encoder_layers = any(k.startswith("encoder_layers.") for k in state_dict)
    has_combo_attn = any("combo_attn" in k for k in state_dict)
    has_cross_attn = any("cross_attn" in k for k in state_dict)

    if has_encoder_layers and has_combo_attn:
        is_enc_dec = True
        model = Double_Decoder(**hparams)
    elif has_encoder_layers and has_cross_attn:
        is_enc_dec = True
        model = StandardEncDec(**hparams)
    elif has_encoder_layers:
        is_enc_dec = True
        model = Double_Decoder(**hparams)
    else:
        is_enc_dec = False
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


# ── OOM-safe batching ─────────────────────────────────────────────────────

def _safe_batch(fn, items, batch_size):
    """Run fn on batches of items, halving batch size on OOM."""
    results = []
    i = 0
    bs = batch_size
    while i < len(items):
        batch = items[i:i + bs]
        try:
            results.extend(fn(batch))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [OOM] Reducing batch size to {bs}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  BATCHED LOG-PROB SCORING (for MC benchmarks)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_log_probs_batch(model, tokenizer, pairs, device, is_enc_dec, batch_size=64,
                        desc=None):
    """Score (context_ids, continuation_ids) pairs in batches.

    Returns: list of float (length-normalized mean log-prob per pair)
    """
    if not pairs:
        return []

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    model_seq_len = getattr(model, "seq_len", 2048)

    if is_enc_dec:
        fn = lambda batch: _log_probs_batch_enc_dec(model, batch, device, bos_id, pad_id, model_seq_len)
    else:
        fn = lambda batch: _log_probs_batch_decoder_only(model, batch, device, bos_id, pad_id)

    # Sort by total length for efficient padding, track original order
    indexed = sorted(enumerate(pairs), key=lambda x: len(x[1][0]) + len(x[1][1]))
    sorted_pairs = [p for _, p in indexed]
    orig_indices = [i for i, _ in indexed]

    pbar = tqdm(total=len(sorted_pairs), desc=desc, leave=False) if desc else None

    sorted_scores = []
    i = 0
    bs = batch_size
    while i < len(sorted_pairs):
        batch = sorted_pairs[i:i + bs]
        try:
            scores = fn(batch)
            sorted_scores.extend(scores)
            i += bs
            if pbar:
                pbar.update(len(batch))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [OOM] Reducing batch size to {bs}")

    if pbar:
        pbar.close()

    # Restore original order
    results = [0.0] * len(pairs)
    for sorted_idx, orig_idx in enumerate(orig_indices):
        results[orig_idx] = sorted_scores[sorted_idx]
    return results


def _log_probs_batch_decoder_only(model, pairs, device, bos_id, pad_id):
    sequences = []
    cont_starts = []
    cont_lens = []

    for ctx_ids, cont_ids in pairs:
        seq = [bos_id] + ctx_ids + cont_ids
        sequences.append(seq)
        cont_starts.append(1 + len(ctx_ids))
        cont_lens.append(len(cont_ids))

    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]
    input_t = torch.tensor(padded, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=input_t)["logits"]

    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    scores = []
    for i in range(len(pairs)):
        start = cont_starts[i]
        n = cont_lens[i]
        cont_t = torch.tensor(pairs[i][1], device=device)
        token_lps = log_probs_all[i, start - 1:start - 1 + n].gather(
            1, cont_t.unsqueeze(1)
        ).squeeze(1)
        scores.append(token_lps.sum().item() / n)

    return scores


def _log_probs_batch_enc_dec(model, pairs, device, bos_id, pad_id, model_seq_len):
    enc_seqs = [[bos_id] + ctx for ctx, _ in pairs]
    dec_seqs = [[bos_id] + cont for _, cont in pairs]
    enc_lens = [len(s) for s in enc_seqs]
    cont_lens = [len(cont) for _, cont in pairs]

    max_enc = max(enc_lens)
    max_dec = max(len(s) for s in dec_seqs)

    padded_enc = [s + [pad_id] * (max_enc - len(s)) for s in enc_seqs]
    padded_dec = [s + [pad_id] * (max_dec - len(s)) for s in dec_seqs]

    enc_t = torch.tensor(padded_enc, device=device)
    dec_t = torch.tensor(padded_dec, device=device)
    blocks = torch.tensor(enc_lens, device=device)

    dec_pos = []
    for el in enc_lens:
        positions = [min(el + j, model_seq_len - 1) for j in range(max_dec)]
        dec_pos.append(positions)
    dec_pos_t = torch.tensor(dec_pos, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(
            encoder_input_ids=enc_t,
            decoder_input_ids=dec_t,
            decoder_input_positions=dec_pos_t,
            blocks=blocks,
            sft=True,
        )["logits"]

    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    scores = []
    for i in range(len(pairs)):
        n = cont_lens[i]
        cont_t = torch.tensor(pairs[i][1], device=device)
        # logits[i, 0] predicts first continuation token (after decoder BOS)
        token_lps = log_probs_all[i, :n].gather(1, cont_t.unsqueeze(1)).squeeze(1)
        scores.append(token_lps.sum().item() / n)

    return scores


def get_enc_dec_predictions_batch(model, tokenizer, pairs, device, batch_size=64, desc=None):
    """Score (context_ids, continuation_ids) pairs via the SFT enc-dec path.

    Returns: list of (log_probs [n], predicted_ids [n]) per pair,
    where n = len(continuation_ids).
    """
    if not pairs:
        return []

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    model_seq_len = getattr(model, "seq_len", 2048)

    # Sort by total length for efficient padding
    indexed = sorted(enumerate(pairs), key=lambda x: len(x[1][0]) + len(x[1][1]))
    sorted_pairs = [p for _, p in indexed]
    orig_indices = [i for i, _ in indexed]

    pbar = tqdm(total=len(sorted_pairs), desc=desc, leave=False) if desc else None

    sorted_results = []
    i = 0
    bs = batch_size
    while i < len(sorted_pairs):
        batch = sorted_pairs[i:i + bs]
        try:
            enc_seqs = [[bos_id] + ctx for ctx, _ in batch]
            dec_seqs = [[bos_id] + cont for _, cont in batch]
            enc_lens = [len(s) for s in enc_seqs]
            cont_lens = [len(cont) for _, cont in batch]

            max_enc = max(enc_lens)
            max_dec = max(len(s) for s in dec_seqs)

            padded_enc = [s + [pad_id] * (max_enc - len(s)) for s in enc_seqs]
            padded_dec = [s + [pad_id] * (max_dec - len(s)) for s in dec_seqs]

            enc_t = torch.tensor(padded_enc, device=device)
            dec_t = torch.tensor(padded_dec, device=device)
            blocks = torch.tensor(enc_lens, device=device)

            dec_pos = []
            for el in enc_lens:
                positions = [min(el + j, model_seq_len - 1) for j in range(max_dec)]
                dec_pos.append(positions)
            dec_pos_t = torch.tensor(dec_pos, device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                       enabled=device.type == "cuda"):
                logits = model(
                    encoder_input_ids=enc_t,
                    decoder_input_ids=dec_t,
                    decoder_input_positions=dec_pos_t,
                    blocks=blocks,
                    sft=True,
                )["logits"]

            logits = logits.float()
            log_probs_all = F.log_softmax(logits, dim=-1)

            for j in range(len(batch)):
                n = cont_lens[j]
                cont_t = torch.tensor(batch[j][1], device=device)
                # logits[j, 0] predicts first continuation token (after decoder BOS)
                token_lps = log_probs_all[j, :n].gather(1, cont_t.unsqueeze(1)).squeeze(1)
                predicted = logits[j, :n].argmax(dim=-1)
                sorted_results.append((token_lps, predicted))

            i += bs
            if pbar:
                pbar.update(len(batch))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [OOM] Reducing batch size to {bs}")

    if pbar:
        pbar.close()

    # Restore original order
    results = [None] * len(pairs)
    for sorted_idx, orig_idx in enumerate(orig_indices):
        results[orig_idx] = sorted_results[sorted_idx]
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  BATCHED SEQUENCE LOG-PROBS (for intrinsic evals + LAMBADA)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_sequence_log_probs_batch(model, tokenizer, id_lists, device, is_enc_dec,
                                 batch_size=64, desc=None):
    """Batched per-token log probs for multiple sequences.

    Returns: list of (log_probs tensor [seq_len-1], predicted_ids tensor [seq_len-1])
    where seq_len is the ORIGINAL length (without any prepended BOS).
    """
    if not id_lists:
        return []

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    # Prepend BOS to sequences that don't already start with one.
    # Training data always starts with BOS; raw text (Wikitext, LAMBADA) doesn't.
    prepended = []
    for ids in id_lists:
        if ids and ids[0] != bos_id:
            prepended.append(True)
        else:
            prepended.append(False)
    id_lists_fixed = [
        [bos_id] + ids if added else ids
        for ids, added in zip(id_lists, prepended)
    ]

    fn = lambda batch: _seq_lp_batch_inner(model, batch, device, pad_id, is_enc_dec)

    # Sort by length for efficient padding
    indexed = sorted(enumerate(id_lists_fixed), key=lambda x: len(x[1]))
    sorted_lists = [ids for _, ids in indexed]
    sorted_prepended = [prepended[i] for i, _ in indexed]
    orig_indices = [i for i, _ in indexed]

    pbar = tqdm(total=len(sorted_lists), desc=desc, leave=False) if desc else None

    sorted_results = []
    i = 0
    bs = batch_size
    while i < len(sorted_lists):
        batch = sorted_lists[i:i + bs]
        try:
            sorted_results.extend(fn(batch))
            i += bs
            if pbar:
                pbar.update(len(batch))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [OOM] Reducing batch size to {bs}")

    if pbar:
        pbar.close()

    # Restore original order and strip the extra BOS position from results
    # where we prepended one. _seq_lp_batch_inner returns (log_probs[seq-1],
    # predicted[seq-1]). If we prepended BOS, the first entry in each tensor
    # corresponds to predicting the original first token — keep that, but the
    # caller's indices assume the original sequence length, so we trim the
    # extra position added by BOS (position 0 predicts BOS→first_token,
    # which is the token the caller considers position 0).
    results = [None] * len(id_lists)
    for sorted_idx, orig_idx in enumerate(orig_indices):
        log_probs, predicted = sorted_results[sorted_idx]
        if sorted_prepended[sorted_idx]:
            # BOS was prepended: position 0 in results predicts the original
            # first token (BOS→tok[0]). The caller expects results indexed to
            # the original sequence where tok[0]→tok[1] is position 0.
            # So skip the first entry (BOS→tok[0] prediction).
            log_probs = log_probs[1:]
            predicted = predicted[1:]
        results[orig_idx] = (log_probs, predicted)
    return results


def _seq_lp_batch_inner(model, id_lists, device, pad_id, is_enc_dec):
    lengths = [len(ids) for ids in id_lists]
    max_len = max(lengths)
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in id_lists]
    input_t = torch.tensor(padded, device=device)

    if is_enc_dec:
        # Each enc-dec arch interprets `blocks` differently — DD wants per-seq
        # boundary positions for combo-attn block masking, SED wants per-batch
        # encoder lengths for create_masks_ED's `enc_lens[b]` indexing. Passing
        # the wrong shape OOB-reads the closure tensor → silent garbage or IMA.
        model_class_name = type(model).__name__
        if model_class_name == "StandardEncDec":
            blocks = torch.full((input_t.shape[0],), max_len,
                                dtype=torch.int64, device=device)
        else:
            blocks = _sample_pretrain_blocks(max_len, device, seed=max_len)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t, blocks=blocks, sft=False)["logits"]
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t)["logits"]

    logits = logits.float()
    log_probs_all = F.log_softmax(logits, dim=-1)

    results = []
    for i, length in enumerate(lengths):
        targets = torch.tensor(id_lists[i][1:], device=device)
        token_lps = log_probs_all[i, :length - 1].gather(1, targets.unsqueeze(1)).squeeze(1)
        predicted = logits[i, :length - 1].argmax(dim=-1)
        results.append((token_lps, predicted))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-EXAMPLE FUNCTIONS (kept for generation evals, NIAH)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_sequence_log_probs(model, tokenizer, input_ids_list, device, is_enc_dec):
    """Per-token log probs for a single packed sequence."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    added_bos = False
    if input_ids_list and input_ids_list[0] != bos_id:
        input_ids_list = [bos_id] + input_ids_list
        added_bos = True

    input_t = torch.tensor([input_ids_list], device=device)

    if is_enc_dec:
        seq_len = len(input_ids_list)
        # Same DD-vs-SED divergence as _seq_lp_batch_inner: SED's create_masks_ED
        # reads `enc_lens[b]` so blocks must be shape [B], not boundary positions.
        model_class_name = type(model).__name__
        if model_class_name == "StandardEncDec":
            blocks = torch.full((1,), seq_len, dtype=torch.int64, device=device)
        else:
            blocks = _sample_pretrain_blocks(seq_len, device, seed=seq_len)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t, blocks=blocks, sft=False)["logits"]
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t)["logits"]

    logits = logits[0].float()
    log_probs = F.log_softmax(logits, dim=-1)

    targets = torch.tensor(input_ids_list[1:], device=device)
    token_lps = log_probs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    predicted = logits[:-1].argmax(dim=-1)

    if added_bos:
        # Strip the BOS→first_token prediction so indices match the original sequence
        token_lps = token_lps[1:]
        predicted = predicted[1:]

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

    # masks=None routes through the layer's flash-attn (or manual SDPA) inference
    # branch, which natively computes causal-self + full-cross — same semantics as
    # create_inference_masks but without rebuilding a BlockMask every decode step.
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.decode(dec_ids, enc_out, None, dec_pos)

        next_tok = _sample_next(logits[:, -1, :], temperature, top_k)
        dec_ids = torch.cat([dec_ids, next_tok], dim=-1)
        dec_pos = torch.cat([dec_pos, dec_pos[:, -1:] + 1], dim=1)

        if next_tok.item() == eos_id:
            break

    gen_ids = dec_ids[0, 1:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
#  BATCHED GREEDY GENERATION (for generation evals)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_text_batch(model, tokenizer, prompts, max_new_tokens, device, is_enc_dec,
                        batch_size=16):
    """Greedy-generate for a list of prompts in batches. Returns list of strings."""
    all_results = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        try:
            if is_enc_dec:
                results = _gen_batch_enc_dec(model, tokenizer, batch_prompts,
                                            max_new_tokens, device)
            else:
                results = _gen_batch_decoder_only(model, tokenizer, batch_prompts,
                                                  max_new_tokens, device)
            all_results.extend(results)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Fallback: generate one at a time for this batch
            for p in batch_prompts:
                all_results.append(generate_text(model, tokenizer, p, max_new_tokens,
                                                 device, is_enc_dec, temperature=0.0))
    return all_results


def _gen_batch_decoder_only(model, tokenizer, prompts, max_new_tokens, device):
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    # Tokenize and left-pad so generation starts at the same position
    all_ids = [[bos_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_prompt_len = max(len(ids) for ids in all_ids)
    prompt_lens = [len(ids) for ids in all_ids]

    # Left-pad: [pad pad pad bos tok tok tok]
    padded = [[pad_id] * (max_prompt_len - len(ids)) + ids for ids in all_ids]
    input_t = torch.tensor(padded, device=device)
    B = input_t.shape[0]

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_t)["logits"][:, -1, :]  # [B, vocab]

        next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1] greedy
        next_tok[finished] = pad_id
        input_t = torch.cat([input_t, next_tok], dim=-1)

        finished = finished | (next_tok.squeeze(-1) == eos_id)
        if finished.all():
            break

    # Extract generated tokens per sequence
    results = []
    for i in range(B):
        gen_start = max_prompt_len  # all prompts left-padded to same length
        gen_ids = input_t[i, gen_start:].tolist()
        # Truncate at EOS or pad
        clean = []
        for t in gen_ids:
            if t == eos_id or t == pad_id:
                break
            clean.append(t)
        results.append(tokenizer.decode(clean, skip_special_tokens=True))

    return results


def _gen_batch_enc_dec(model, tokenizer, prompts, max_new_tokens, device):
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    assistant_id = tokenizer.convert_tokens_to_ids("<assistant>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    # Tokenize encoder inputs and right-pad
    all_enc_ids = [[bos_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    enc_lens = [len(ids) for ids in all_enc_ids]
    max_enc_len = max(enc_lens)
    padded_enc = [ids + [pad_id] * (max_enc_len - len(ids)) for ids in all_enc_ids]
    enc_input = torch.tensor(padded_enc, device=device)
    B = enc_input.shape[0]

    # Encode all prompts at once
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        enc_out = model.encode(enc_input)  # [B, max_enc_len, dim]

    # Initialize decoder: all start with assistant token
    dec_ids = torch.full((B, 1), assistant_id, device=device, dtype=torch.long)
    # Decoder positions: each starts at its own enc_len
    dec_pos = torch.tensor([[el] for el in enc_lens], device=device, dtype=torch.long)

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # masks=None routes through the layer's flash-attn (or manual SDPA) inference
    # branch — see _gen_enc_dec for the rationale.
    for step in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.decode(dec_ids, enc_out, None, dec_pos)

        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1] greedy
        next_tok[finished] = pad_id
        dec_ids = torch.cat([dec_ids, next_tok], dim=-1)
        dec_pos = torch.cat([dec_pos, dec_pos[:, -1:] + 1], dim=1)

        finished = finished | (next_tok.squeeze(-1) == eos_id)
        if finished.all():
            break

    # Extract per-sequence
    results = []
    for i in range(B):
        gen_ids = dec_ids[i, 1:].tolist()  # skip assistant token
        clean = []
        for t in gen_ids:
            if t == eos_id or t == pad_id:
                break
            clean.append(t)
        results.append(tokenizer.decode(clean, skip_special_tokens=True))

    return results
