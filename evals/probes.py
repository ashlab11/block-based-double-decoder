"""
Synthetic probes: Needle in a Haystack and Copy/Retrieval tasks.
These test the model's ability to attend to and retrieve specific
information from context — directly relevant to the combo attention mechanism.
"""

import random
import torch
from tqdm import tqdm
from evals.utils import generate_text


# ═══════════════════════════════════════════════════════════════════════════
#  Needle in a Haystack
# ═══════════════════════════════════════════════════════════════════════════

# Filler passages for the haystack (generic text)
_FILLER_SENTENCES = [
    "The weather was pleasant and the birds were singing in the trees.",
    "Many researchers have studied the effects of climate on agriculture.",
    "The history of mathematics spans thousands of years across many cultures.",
    "Advances in technology have transformed how people communicate globally.",
    "The ocean covers more than seventy percent of the Earth's surface.",
    "Classical music has a rich tradition dating back several centuries.",
    "Photography was invented in the early nineteenth century.",
    "The study of languages reveals much about human cognition.",
    "Mountains form through tectonic plate collisions over millions of years.",
    "Libraries have served as centers of learning for thousands of years.",
    "The periodic table organizes elements by atomic number and properties.",
    "Ancient civilizations developed complex systems of writing and governance.",
    "Modern transportation networks connect cities across continents.",
    "The human brain contains approximately eighty-six billion neurons.",
    "Forests play a critical role in regulating the global carbon cycle.",
    "Architecture reflects the cultural values of the society that creates it.",
]

# Distinctive needle facts that won't appear in filler
_NEEDLES = [
    ("The secret code for the vault is 7-4-2-9-1.", "What is the secret code for the vault?", "7-4-2-9-1"),
    ("The capital of Zandoria is Felverton.", "What is the capital of Zandoria?", "Felverton"),
    ("Dr. Marquez won the Nobel Prize in 2037.", "Who won the Nobel Prize in 2037?", "Dr. Marquez"),
    ("The password to the server room is blue-tiger-42.", "What is the password to the server room?", "blue-tiger-42"),
    ("Project Artemis launched on March 15, 2031.", "When did Project Artemis launch?", "March 15, 2031"),
]


def _build_haystack(needle_text, position_frac, target_tokens, tokenizer):
    """Build a haystack with the needle inserted at position_frac (0.0=start, 1.0=end)."""
    rng = random.Random(42)

    # Build filler text until we reach target token count
    filler_parts = []
    current_tokens = 0
    while current_tokens < target_tokens:
        sentence = rng.choice(_FILLER_SENTENCES)
        filler_parts.append(sentence)
        current_tokens += len(tokenizer.encode(sentence, add_special_tokens=False))

    # Insert needle at the specified position
    insert_idx = max(1, int(len(filler_parts) * position_frac))
    filler_parts.insert(insert_idx, needle_text)

    return " ".join(filler_parts)


def eval_needle_in_haystack(model, tokenizer, device, is_enc_dec,
                            context_lengths=None, depth_fractions=None,
                            max_gen_tokens=32):
    """Needle in a Haystack: insert a fact into filler text, ask about it.

    Tests at multiple context lengths and needle depths.
    Returns a grid of retrieval accuracy.
    """
    if context_lengths is None:
        seq_len = getattr(model, "seq_len", 2048)
        # Test at 25%, 50%, 75%, 100% of max context
        context_lengths = [seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 64]

    if depth_fractions is None:
        depth_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

    results_grid = []
    total_correct = total_tests = 0

    for ctx_len in tqdm(context_lengths, desc="NIAH context lengths"):
        for depth in depth_fractions:
            needle_text, question, answer = random.Random(42).choice(_NEEDLES)
            haystack = _build_haystack(needle_text, depth, ctx_len, tokenizer)

            prompt = haystack + f"\n\nQuestion: {question}\nAnswer:"
            # Truncate prompt to fit context
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > ctx_len:
                prompt_ids = prompt_ids[:ctx_len]
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)

            generated = generate_text(model, tokenizer, prompt, max_gen_tokens,
                                      device, is_enc_dec, temperature=0.0)

            found = answer.lower() in generated.lower()
            total_correct += int(found)
            total_tests += 1

            results_grid.append({
                "context_length": ctx_len,
                "depth_fraction": depth,
                "found": found,
                "generated": generated[:100],
                "expected": answer,
            })

    return {
        "name": "Needle in Haystack",
        "accuracy": total_correct / max(total_tests, 1),
        "total": total_tests,
        "grid": results_grid,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Copy / Retrieval Synthetic Task
# ═══════════════════════════════════════════════════════════════════════════

def _make_copy_example(seq_len, vocab_size, repeat_offset, rng):
    """Create a sequence A B C ... [SEP] ... A B C ... [SEP] A B ?
    where the model must predict C (copying from earlier context).

    Returns: (full_ids, target_token, prediction_position)
    """
    # Random pattern to copy
    pattern_len = rng.randint(3, 8)
    pattern = [rng.randint(10, vocab_size - 1) for _ in range(pattern_len)]

    # Random filler between patterns
    filler_len = max(4, repeat_offset - pattern_len)
    filler = [rng.randint(10, vocab_size - 1) for _ in range(filler_len)]

    # Build: pattern + filler + pattern[:-1]  (model must predict pattern[-1])
    sequence = pattern + filler + pattern[:-1]

    target = pattern[-1]
    pred_pos = len(sequence) - 1  # position of last token, model predicts next

    return sequence, target, pred_pos


def eval_copy_retrieval(model, tokenizer, device, is_enc_dec,
                        num_examples=200, max_offset=512, batch_size=64):
    """Test exact token copying from earlier in the sequence (batched per offset)."""
    vocab_size = tokenizer.vocab_size
    rng = random.Random(42)

    offsets = [32, 64, 128, 256, min(max_offset, 512)]
    examples_per_offset = max(num_examples // len(offsets), 10)

    results_by_offset = {}
    total_correct = total_tests = 0
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    for offset in tqdm(offsets, desc="Copy/Retrieval"):
        # Generate all examples for this offset
        seqs, targets, pred_positions = [], [], []
        for _ in range(examples_per_offset):
            seq, target, pred_pos = _make_copy_example(
                seq_len=offset + 20, vocab_size=vocab_size,
                repeat_offset=offset, rng=rng
            )
            seqs.append(seq)
            targets.append(target)
            pred_positions.append(pred_pos)

        # Batch forward pass
        correct = 0
        for batch_start in range(0, len(seqs), batch_size):
            batch_seqs = seqs[batch_start:batch_start + batch_size]
            batch_targets = targets[batch_start:batch_start + batch_size]
            batch_ppos = pred_positions[batch_start:batch_start + batch_size]

            max_len = max(len(s) for s in batch_seqs)
            padded = [s + [pad_id] * (max_len - len(s)) for s in batch_seqs]
            input_t = torch.tensor(padded, device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                      enabled=device.type == "cuda"):
                if is_enc_dec:
                    blocks = torch.tensor([max_len // 2], device=device)
                    logits = model(input_ids=input_t, blocks=blocks, sft=False)["logits"]
                else:
                    logits = model(input_ids=input_t)["logits"]

            for i in range(len(batch_seqs)):
                predicted = logits[i, batch_ppos[i]].argmax().item()
                correct += int(predicted == batch_targets[i])

        count = len(seqs)
        acc = correct / max(count, 1)
        results_by_offset[offset] = {"accuracy": acc, "correct": correct, "total": count}
        total_correct += correct
        total_tests += count

    return {
        "name": "Copy/Retrieval",
        "accuracy": total_correct / max(total_tests, 1),
        "by_offset": results_by_offset,
        "total": total_tests,
    }
