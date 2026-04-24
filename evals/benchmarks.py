"""
Log-likelihood benchmarks: multiple-choice evaluations, LAMBADA,
GLUE, and SuperGLUE formatted for generative scoring.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from evals.utils import score_choices, get_log_probs, get_sequence_log_probs


# ── Generic MC evaluator ──────────────────────────────────────────────────

def _eval_mc(name, model, tokenizer, device, is_enc_dec,
             dataset_args, format_fn, max_examples=None):
    """Run a multiple-choice benchmark.

    format_fn(example) -> (context_str, [choices_str], gold_idx)
    """
    ds = load_dataset(**dataset_args)
    if max_examples and not hasattr(ds, '__iter__'):
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = total = 0
    for ex in tqdm(ds, desc=name):
        if max_examples and total >= max_examples:
            break
        try:
            context, choices, gold = format_fn(ex)
            pred, _ = score_choices(model, tokenizer, context, choices, device, is_enc_dec)
            correct += int(pred == gold)
            total += 1
        except Exception as e:
            print(f"  [{name}] Skipped example: {e}")

    return {"name": name, "accuracy": correct / max(total, 1),
            "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════════════════
#  COMMONSENSE / REASONING BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

# ── HellaSwag ──────────────────────────────────────────────────────────────

def _fmt_hellaswag(ex):
    ctx = ex["activity_label"] + ". " + ex["ctx_a"] + " " + ex["ctx_b"]
    return ctx, ex["endings"], int(ex["label"])

def eval_hellaswag(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("HellaSwag", model, tokenizer, device, is_enc_dec,
                    {"path": "Rowan/hellaswag", "split": "validation"},
                    _fmt_hellaswag, max_examples)


# ── PIQA ───────────────────────────────────────────────────────────────────

def _fmt_piqa(ex):
    return ex["goal"], [" " + ex["sol1"], " " + ex["sol2"]], ex["label"]

def eval_piqa(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("PIQA", model, tokenizer, device, is_enc_dec,
                    {"path": "MillerLab/piqa", "split": "validation"},
                    _fmt_piqa, max_examples)


# ── ARC-Easy / ARC-Challenge ──────────────────────────────────────────────

def _fmt_arc(ex):
    ctx = "Question: " + ex["question"] + "\nAnswer:"
    choices = [" " + c for c in ex["choices"]["text"]]
    labels = ex["choices"]["label"]
    gold = labels.index(ex["answerKey"])
    return ctx, choices, gold

def eval_arc_easy(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("ARC-Easy", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/ai2_arc", "name": "ARC-Easy", "split": "test"},
                    _fmt_arc, max_examples)

def eval_arc_challenge(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("ARC-Challenge", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test"},
                    _fmt_arc, max_examples)


# ── WinoGrande ────────────────────────────────────────────────────────────

def _fmt_winogrande(ex):
    parts = ex["sentence"].split("_")
    ctx = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    choices = [ex["option1"] + rest, ex["option2"] + rest]
    gold = int(ex["answer"]) - 1  # "1"/"2" → 0/1
    return ctx, choices, gold

def eval_winogrande(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("WinoGrande", model, tokenizer, device, is_enc_dec,
                    {"path": "allenai/winogrande", "name": "winogrande_xl",
                     "split": "validation"},
                    _fmt_winogrande, max_examples)


# ── BoolQ ──────────────────────────────────────────────────────────────────

def _fmt_boolq(ex):
    ctx = ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:"
    return ctx, [" No", " Yes"], int(ex["answer"])

def eval_boolq(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("BoolQ", model, tokenizer, device, is_enc_dec,
                    {"path": "google/boolq", "split": "validation"},
                    _fmt_boolq, max_examples)


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

def eval_mmlu(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("MMLU", model, tokenizer, device, is_enc_dec,
                    {"path": "cais/mmlu", "name": "all", "split": "test"},
                    _fmt_mmlu, max_examples)


# ── TruthfulQA (MC1) ──────────────────────────────────────────────────────

def _fmt_truthfulqa(ex):
    ctx = "Q: " + ex["question"] + "\nA:"
    choices = [" " + c for c in ex["mc1_targets"]["choices"]]
    gold = ex["mc1_targets"]["labels"].index(1)
    return ctx, choices, gold

def eval_truthfulqa(model, tokenizer, device, is_enc_dec, max_examples=None):
    return _eval_mc("TruthfulQA-MC1", model, tokenizer, device, is_enc_dec,
                    {"path": "truthfulqa/truthful_qa", "name": "multiple_choice",
                     "split": "validation"},
                    _fmt_truthfulqa, max_examples)


# ═══════════════════════════════════════════════════════════════════════════
#  LAMBADA (last-word prediction — exact match, not MC)
# ═══════════════════════════════════════════════════════════════════════════

def eval_lambada(model, tokenizer, device, is_enc_dec, max_examples=None):
    """LAMBADA: predict the final word of a passage. Greedy exact match."""
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = total = 0
    for ex in tqdm(ds, desc="LAMBADA"):
        text = ex["text"]
        idx = text.rfind(" ")
        if idx < 0:
            continue
        context = text[:idx]
        target = text[idx:]  # includes leading space

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            continue

        # Get model predictions via full-sequence forward
        full_ids = context_ids + target_ids
        _, predicted = get_sequence_log_probs(model, tokenizer, full_ids, device, is_enc_dec)

        # Check greedy match for each target token
        all_correct = True
        for i, tid in enumerate(target_ids):
            pos = len(context_ids) - 1 + i  # logits[pos] predicts full_ids[pos+1]
            if predicted[pos].item() != tid:
                all_correct = False
                break

        correct += int(all_correct)
        total += 1

    return {"name": "LAMBADA", "accuracy": correct / max(total, 1),
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
            ex["label"],  # 0=entailment(Yes), 1=not(No)
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


def eval_glue(model, tokenizer, device, is_enc_dec, max_examples=None):
    """Run all GLUE subtasks. Returns per-task and average accuracy."""
    results = {}
    accs = []
    for task_name, cfg in _GLUE_TASKS.items():
        r = _eval_mc(f"GLUE/{task_name}", model, tokenizer, device, is_enc_dec,
                     cfg["dataset"], cfg["format"], max_examples)
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


def eval_superglue(model, tokenizer, device, is_enc_dec, max_examples=None):
    """Run all SuperGLUE subtasks. Returns per-task and average accuracy."""
    results = {}
    accs = []
    for task_name, cfg in _SUPERGLUE_TASKS.items():
        r = _eval_mc(f"SuperGLUE/{task_name}", model, tokenizer, device, is_enc_dec,
                     cfg["dataset"], cfg["format"], max_examples)
        results[task_name] = r
        accs.append(r["accuracy"])

    results["average"] = sum(accs) / len(accs)
    return {"name": "SuperGLUE", "subtasks": results, "average": results["average"]}
