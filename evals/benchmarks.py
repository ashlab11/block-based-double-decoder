"""
Log-likelihood benchmarks: multiple-choice evaluations, LAMBADA,
GLUE, and SuperGLUE formatted for generative scoring.

All MC benchmarks use batched GPU scoring for maximum throughput.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from evals.utils import get_log_probs_batch, get_sequence_log_probs_batch, get_enc_dec_predictions_batch


# ── Generic batched MC evaluator ─────────────────────────────────────────

def _eval_mc(name, model, tokenizer, device, is_enc_dec,
             dataset_args, format_fn, max_examples=None, batch_size=64):
    """Run a multiple-choice benchmark with batched scoring.

    format_fn(example) -> (context_str, [choices_str], gold_idx)
    """
    ds = load_dataset(**dataset_args)

    # 1. Collect and format all examples
    examples = []
    for ex in ds:
        if max_examples and len(examples) >= max_examples:
            break
        try:
            context, choices, gold = format_fn(ex)
            examples.append((context, choices, gold))
        except Exception:
            continue

    if not examples:
        return {"name": name, "accuracy": 0, "correct": 0, "total": 0}

    # 2. Tokenize and flatten to (context_ids, choice_ids) pairs
    all_pairs = []
    num_choices_per = []

    for ctx_str, choices, _ in examples:
        ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)
        n = 0
        for ch_str in choices:
            ch_ids = tokenizer.encode(ch_str, add_special_tokens=False)
            if ch_ids:
                all_pairs.append((ctx_ids, ch_ids))
                n += 1
        num_choices_per.append(n)

    # 3. Batch score all pairs
    all_scores = get_log_probs_batch(
        model, tokenizer, all_pairs, device, is_enc_dec,
        batch_size=batch_size, desc=name
    )

    # 4. Aggregate per example
    correct = total = 0
    score_idx = 0
    for ex_idx, (_, choices, gold) in enumerate(examples):
        n = num_choices_per[ex_idx]
        scores = all_scores[score_idx:score_idx + n]
        score_idx += n
        pred = max(range(n), key=lambda i: scores[i])
        correct += int(pred == gold)
        total += 1

    return {"name": name, "accuracy": correct / max(total, 1),
            "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════════════════
#  COMMONSENSE / REASONING BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

# ── HellaSwag ──────────────────────────────────────────────────────────────

def _fmt_hellaswag(ex):
    ctx = ex["activity_label"] + ". " + ex["ctx_a"] + " " + ex["ctx_b"]
    return ctx, ex["endings"], int(ex["label"])

def eval_hellaswag(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("HellaSwag", model, tokenizer, device, is_enc_dec,
                    {"path": "Rowan/hellaswag", "split": "validation"},
                    _fmt_hellaswag, max_examples, batch_size)


# ── PIQA ───────────────────────────────────────────────────────────────────

def _fmt_piqa(ex):
    return ex["goal"], [" " + ex["sol1"], " " + ex["sol2"]], ex["label"]

def eval_piqa(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("PIQA", model, tokenizer, device, is_enc_dec,
                    {"path": "lighteval/piqa", "split": "validation"},
                    _fmt_piqa, max_examples, batch_size)


# ── ARC-Easy / ARC-Challenge ──────────────────────────────────────────────

def _fmt_arc(ex):
    ctx = "Question: " + ex["question"] + "\nAnswer:"
    choices = [" " + c for c in ex["choices"]["text"]]
    labels = ex["choices"]["label"]
    gold = labels.index(ex["answerKey"])
    return ctx, choices, gold

def eval_arc_easy(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("ARC-Easy", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/ai2_arc", "name": "ARC-Easy", "split": "test"},
                    _fmt_arc, max_examples, batch_size)

def eval_arc_challenge(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("ARC-Challenge", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test"},
                    _fmt_arc, max_examples, batch_size)


# ── WinoGrande ────────────────────────────────────────────────────────────

def _fmt_winogrande(ex):
    parts = ex["sentence"].split("_")
    ctx = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    choices = [ex["option1"] + rest, ex["option2"] + rest]
    gold = int(ex["answer"]) - 1  # "1"/"2" → 0/1
    return ctx, choices, gold

def eval_winogrande(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("WinoGrande", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/winogrande", "name": "winogrande_xl",
                     "split": "validation"},
                    _fmt_winogrande, max_examples, batch_size)


# ── BoolQ ──────────────────────────────────────────────────────────────────

def _fmt_boolq(ex):
    ctx = ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:"
    return ctx, [" No", " Yes"], int(ex["answer"])

def eval_boolq(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("BoolQ", model, tokenizer, device, is_enc_dec,
                    {"path": "google/boolq", "split": "validation"},
                    _fmt_boolq, max_examples, batch_size)


# ── MMLU ───────────────────────────────────────────────────────────────────

def _fmt_mmlu(ex):
    letters = ["A", "B", "C", "D"]
    q = ex["question"] + "\n"
    for i, choice in enumerate(ex["choices"]):
        q += f"{letters[i]}. {choice}\n"
    q += "Answer:"
    label_choices = [" " + l for l in letters]
    gold = ex["answer"]  # 0-3
    return q, label_choices, gold

def eval_mmlu(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("MMLU", model, tokenizer, device, is_enc_dec,
                    {"path": "cais/mmlu", "name": "all", "split": "test"},
                    _fmt_mmlu, max_examples, batch_size)


# ── TruthfulQA (MC1) ──────────────────────────────────────────────────────

def _fmt_truthfulqa(ex):
    ctx = "Q: " + ex["question"] + "\nA:"
    choices = [" " + c for c in ex["mc1_targets"]["choices"]]
    gold = ex["mc1_targets"]["labels"].index(1)
    return ctx, choices, gold

def eval_truthfulqa(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    return _eval_mc("TruthfulQA-MC1", model, tokenizer, device, is_enc_dec,
                    {"path": "truthfulqa/truthful_qa", "name": "multiple_choice",
                     "split": "validation"},
                    _fmt_truthfulqa, max_examples, batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  LAMBADA (last-word prediction — batched exact match)
# ═══════════════════════════════════════════════════════════════════════════

def eval_lambada(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    """LAMBADA: predict the final word of a passage. Greedy exact match."""
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Collect all examples
    examples = []
    for ex in ds:
        text = ex["text"]
        idx = text.rfind(" ")
        if idx < 0:
            continue
        context_ids = tokenizer.encode(text[:idx], add_special_tokens=False)
        target_ids = tokenizer.encode(text[idx:], add_special_tokens=False)
        if target_ids:
            examples.append((context_ids, target_ids))

    # Batch score full sequences
    all_id_lists = [ctx + tgt for ctx, tgt in examples]
    all_results = get_sequence_log_probs_batch(
        model, tokenizer, all_id_lists, device, is_enc_dec,
        batch_size=batch_size, desc="LAMBADA"
    )

    correct = 0
    for i, (ctx_ids, tgt_ids) in enumerate(examples):
        _, predicted = all_results[i]
        all_match = True
        for j, tid in enumerate(tgt_ids):
            pos = len(ctx_ids) - 1 + j
            if pos >= len(predicted) or predicted[pos].item() != tid:
                all_match = False
                break
        correct += int(all_match)

    total = len(examples)
    return {"name": "LAMBADA", "accuracy": correct / max(total, 1),
            "correct": correct, "total": total}


def eval_lambada_enc_dec(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    """LAMBADA via enc-dec SFT path: context→encoder, last word→decoder.

    This evaluates the model the way it's designed to be used — the full
    context passage goes to the encoder, and the target word is scored
    from the decoder with cross-attention to the encoder output.

    Falls back to the pretrain path for decoder-only models.
    """
    if not is_enc_dec:
        return eval_lambada(model, tokenizer, device, is_enc_dec, max_examples, batch_size)

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Collect (context_ids, target_ids) pairs
    pairs = []
    for ex in ds:
        text = ex["text"]
        idx = text.rfind(" ")
        if idx < 0:
            continue
        context_ids = tokenizer.encode(text[:idx], add_special_tokens=False)
        target_ids = tokenizer.encode(text[idx:], add_special_tokens=False)
        if target_ids:
            pairs.append((context_ids, target_ids))

    # Score via SFT enc-dec path (context→encoder, target→decoder)
    all_results = get_enc_dec_predictions_batch(
        model, tokenizer, pairs, device,
        batch_size=batch_size, desc="LAMBADA (enc-dec)"
    )

    correct = 0
    for i, (ctx_ids, tgt_ids) in enumerate(pairs):
        _, predicted = all_results[i]
        tgt_t = torch.tensor(tgt_ids, device=predicted.device)
        n = min(len(tgt_ids), len(predicted))
        if n == len(tgt_ids) and (predicted[:n] == tgt_t[:n]).all():
            correct += 1

    total = len(pairs)
    return {"name": "LAMBADA (enc-dec)", "accuracy": correct / max(total, 1),
            "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════════════════
#  GLUE (formatted as log-likelihood scoring)
# ═══════════════════════════════════════════════════════════════════════════

_GLUE_TASKS = {
    "sst2": {
        "dataset": {"path": "nyu-mll/glue", "name": "sst2", "split": "validation"},
        "format": lambda ex: (
            ex["sentence"] + "\nSentiment:",
            [" negative", " positive"],
            ex["label"],
        ),
    },
    "mnli": {
        "dataset": {"path": "nyu-mll/glue", "name": "mnli", "split": "validation_matched"},
        "format": lambda ex: (
            "Premise: " + ex["premise"] + "\nHypothesis: " + ex["hypothesis"]
            + "\nRelation:",
            [" entailment", " neutral", " contradiction"],
            ex["label"],
        ),
    },
    "qqp": {
        "dataset": {"path": "nyu-mll/glue", "name": "qqp", "split": "validation"},
        "format": lambda ex: (
            "Question 1: " + ex["question1"] + "\nQuestion 2: " + ex["question2"]
            + "\nDuplicate?",
            [" No", " Yes"],
            ex["label"],
        ),
    },
    "qnli": {
        "dataset": {"path": "nyu-mll/glue", "name": "qnli", "split": "validation"},
        "format": lambda ex: (
            "Question: " + ex["question"] + "\n" + ex["sentence"]
            + "\nDoes this answer the question?",
            [" Yes", " No"],
            ex["label"],
        ),
    },
    "rte": {
        "dataset": {"path": "nyu-mll/glue", "name": "rte", "split": "validation"},
        "format": lambda ex: (
            ex["sentence1"] + "\n" + ex["sentence2"] + "\nTrue or False?",
            [" True", " False"],
            ex["label"],
        ),
    },
    "mrpc": {
        "dataset": {"path": "nyu-mll/glue", "name": "mrpc", "split": "validation"},
        "format": lambda ex: (
            "Sentence 1: " + ex["sentence1"] + "\nSentence 2: " + ex["sentence2"]
            + "\nParaphrase?",
            [" No", " Yes"],
            ex["label"],
        ),
    },
    "cola": {
        "dataset": {"path": "nyu-mll/glue", "name": "cola", "split": "validation"},
        "format": lambda ex: (
            ex["sentence"] + "\nGrammatically acceptable?",
            [" No", " Yes"],
            ex["label"],
        ),
    },
}


def eval_glue(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    """Run all GLUE subtasks with batched scoring."""
    results = {}
    accs = []
    for task_name, cfg in _GLUE_TASKS.items():
        r = _eval_mc(f"GLUE/{task_name}", model, tokenizer, device, is_enc_dec,
                     cfg["dataset"], cfg["format"], max_examples, batch_size)
        results[task_name] = r
        accs.append(r["accuracy"])

    results["average"] = sum(accs) / len(accs)
    return {"name": "GLUE", "subtasks": results, "average": results["average"]}


# ═══════════════════════════════════════════════════════════════════════════
#  SuperGLUE (formatted as log-likelihood scoring)
# ═══════════════════════════════════════════════════════════════════════════

_SUPERGLUE_TASKS = {
    "boolq": {
        "dataset": {"path": "super_glue", "name": "boolq", "split": "validation"},
        "format": lambda ex: (
            ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:",
            [" No", " Yes"],
            ex["label"],
        ),
    },
    "cb": {
        "dataset": {"path": "super_glue", "name": "cb", "split": "validation"},
        "format": lambda ex: (
            "Premise: " + ex["premise"] + "\nHypothesis: " + ex["hypothesis"]
            + "\nRelation:",
            [" entailment", " contradiction", " neutral"],
            ex["label"],
        ),
    },
    "copa": {
        "dataset": {"path": "super_glue", "name": "copa", "split": "validation"},
        "format": lambda ex: (
            ex["premise"] + (" because" if ex["question"] == "cause" else " so"),
            [" " + ex["choice1"], " " + ex["choice2"]],
            ex["label"],
        ),
    },
    "rte": {
        "dataset": {"path": "super_glue", "name": "rte", "split": "validation"},
        "format": lambda ex: (
            ex["premise"] + "\n" + ex["hypothesis"] + "\nTrue or False?",
            [" True", " False"],
            ex["label"],
        ),
    },
    "wic": {
        "dataset": {"path": "super_glue", "name": "wic", "split": "validation"},
        "format": lambda ex: (
            ex["word"] + ": " + ex["sentence1"] + "\n"
            + ex["word"] + ": " + ex["sentence2"] + "\nSame meaning?",
            [" No", " Yes"],
            ex["label"],
        ),
    },
    "wsc.fixed": {
        "dataset": {"path": "super_glue", "name": "wsc.fixed", "split": "validation"},
        "format": lambda ex: (
            ex["text"] + "\nDoes '" + ex["span2_text"] + "' refer to '"
            + ex["span1_text"] + "'?",
            [" No", " Yes"],
            ex["label"],
        ),
    },
}


def eval_superglue(model, tokenizer, device, is_enc_dec, max_examples=None, batch_size=64):
    """Run all SuperGLUE subtasks with batched scoring."""
    results = {}
    accs = []
    for task_name, cfg in _SUPERGLUE_TASKS.items():
        r = _eval_mc(f"SuperGLUE/{task_name}", model, tokenizer, device, is_enc_dec,
                     cfg["dataset"], cfg["format"], max_examples, batch_size)
        results[task_name] = r
        accs.append(r["accuracy"])

    results["average"] = sum(accs) / len(accs)
    return {"name": "SuperGLUE", "subtasks": results, "average": results["average"]}
