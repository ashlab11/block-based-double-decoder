"""
Generation-based evaluations: XSum (ROUGE), SQuAD (F1/EM),
TriviaQA (EM), and HumanEval (pass@k via code execution).
"""

import re
import string
import subprocess
import tempfile
import textwrap
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from evals.utils import generate_text


# ── Text normalization for QA ─────────────────────────────────────────────

def _normalize(text):
    """Normalize for QA matching: lowercase, strip articles/punctuation/whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _f1(prediction, ground_truth):
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not gold_tokens or not pred_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec = num_same / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def _exact_match(prediction, ground_truths):
    pred_norm = _normalize(prediction)
    return float(any(_normalize(g) == pred_norm for g in ground_truths))


def _max_f1(prediction, ground_truths):
    return max(_f1(prediction, g) for g in ground_truths)


# ═══════════════════════════════════════════════════════════════════════════
#  XSum — Abstractive Summarization
# ═══════════════════════════════════════════════════════════════════════════

def _rouge_n(prediction, reference, n=1):
    """Simple ROUGE-N (unigram/bigram recall) without external dependencies."""
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens:
        return 0.0

    pred_ng = Counter(ngrams(pred_tokens, n))
    ref_ng = Counter(ngrams(ref_tokens, n))

    overlap = sum((pred_ng & ref_ng).values())
    total = sum(ref_ng.values())
    return overlap / max(total, 1)


def _rouge_l(prediction, reference):
    """ROUGE-L via longest common subsequence."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def eval_xsum(model, tokenizer, device, is_enc_dec, max_examples=200, max_gen_tokens=64):
    """XSum abstractive summarization. Reports ROUGE-1, ROUGE-2, ROUGE-L."""
    ds = load_dataset("EdinburghNLP/xsum", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    r1_scores, r2_scores, rl_scores = [], [], []

    for ex in tqdm(ds, desc="XSum"):
        prompt = "Summarize the following article in one sentence.\n\nArticle: " \
                 + ex["document"][:1500] + "\n\nSummary:"
        generated = generate_text(model, tokenizer, prompt, max_gen_tokens,
                                  device, is_enc_dec, temperature=0.0)
        reference = ex["summary"]

        r1_scores.append(_rouge_n(generated, reference, 1))
        r2_scores.append(_rouge_n(generated, reference, 2))
        rl_scores.append(_rouge_l(generated, reference))

    return {
        "name": "XSum",
        "rouge1": sum(r1_scores) / max(len(r1_scores), 1),
        "rouge2": sum(r2_scores) / max(len(r2_scores), 1),
        "rougeL": sum(rl_scores) / max(len(rl_scores), 1),
        "total": len(r1_scores),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SQuAD — Extractive Question Answering
# ═══════════════════════════════════════════════════════════════════════════

def eval_squad(model, tokenizer, device, is_enc_dec, max_examples=500, max_gen_tokens=32):
    """SQuAD v1.1 extractive QA. Reports EM and F1."""
    ds = load_dataset("rajpurkar/squad", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    em_total = f1_total = 0.0
    count = 0

    for ex in tqdm(ds, desc="SQuAD"):
        context = ex["context"][:1200]
        question = ex["question"]
        gold_answers = ex["answers"]["text"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        generated = generate_text(model, tokenizer, prompt, max_gen_tokens,
                                  device, is_enc_dec, temperature=0.0)
        generated = generated.strip().split("\n")[0]  # take first line

        em_total += _exact_match(generated, gold_answers)
        f1_total += _max_f1(generated, gold_answers)
        count += 1

    return {
        "name": "SQuAD",
        "exact_match": em_total / max(count, 1),
        "f1": f1_total / max(count, 1),
        "total": count,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TriviaQA — Factual Knowledge Recall
# ═══════════════════════════════════════════════════════════════════════════

def eval_triviaqa(model, tokenizer, device, is_enc_dec, max_examples=500, max_gen_tokens=32):
    """TriviaQA (unfiltered, no context). Reports EM and F1."""
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    em_total = f1_total = 0.0
    count = 0

    for ex in tqdm(ds, desc="TriviaQA"):
        question = ex["question"]
        gold_answers = ex["answer"]["aliases"] + [ex["answer"]["value"]]

        prompt = f"Question: {question}\nAnswer:"
        generated = generate_text(model, tokenizer, prompt, max_gen_tokens,
                                  device, is_enc_dec, temperature=0.0)
        generated = generated.strip().split("\n")[0]

        em_total += _exact_match(generated, gold_answers)
        f1_total += _max_f1(generated, gold_answers)
        count += 1

    return {
        "name": "TriviaQA",
        "exact_match": em_total / max(count, 1),
        "f1": f1_total / max(count, 1),
        "total": count,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  HumanEval — Code Generation (pass@1 via execution)
# ═══════════════════════════════════════════════════════════════════════════

def _run_code_safely(code, timeout=5):
    """Execute Python code in a subprocess, return True if it passes."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def eval_humaneval(model, tokenizer, device, is_enc_dec, max_examples=None, max_gen_tokens=256):
    """HumanEval: generate function bodies and test via execution. Reports pass@1."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    passed = total = 0
    for ex in tqdm(ds, desc="HumanEval"):
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        generated = generate_text(model, tokenizer, prompt, max_gen_tokens,
                                  device, is_enc_dec, temperature=0.0)

        # Build complete program: prompt + generated body + tests
        # Stop at the next function definition or class
        lines = generated.split("\n")
        body_lines = []
        for line in lines:
            if body_lines and (line.startswith("def ") or line.startswith("class ")):
                break
            body_lines.append(line)
        generated_body = "\n".join(body_lines)

        full_code = prompt + generated_body + "\n" + test_code + f"\ncheck({entry_point})\n"

        if _run_code_safely(full_code):
            passed += 1
        total += 1

    return {
        "name": "HumanEval",
        "pass_at_1": passed / max(total, 1),
        "passed": passed,
        "total": total,
    }
