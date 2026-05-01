from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC, Sequence, Strip
import os
import argparse


def create_tokenizer(vocab_size=32768, corpus_file="data/Pretrain/slimpajama.jsonl", output_file="tokenizer/tokenizer_32k.json"):
    # 1) Initialize a BPE tokenizer with an unk token
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2) Normalization: NFKC + strip
    tokenizer.normalizer = Sequence([NFKC(), Strip()])

    # 3) Pre-tokenization: byte-level (whitespace+punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    special_tokens = ["<unk>", "<s>", "</s>",
        "<user>", "<assistant>", "<pad>", *[f"<sentinel_{i}>" for i in range(15)]
    ]

    # 4) Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )

    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    tokenizer.train([corpus_file], trainer)

    # 5) Decoder
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(output_file)
    print(f"Tokenizer saved to {output_file} (vocab_size={vocab_size})")
    return tokenizer


def test_tokenizer(tokenizer):
    """Test the tokenizer with various inputs"""
    print("\nTesting special token encoding:")
    test_inputs = ["<user>", "<assistant>", "Hello, world!"]
    for text in test_inputs:
        output = tokenizer.encode(text)
        print(f"{text} -> ids={output.ids}, tokens={output.tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--corpus", type=str, default="data/Pretrain/slimpajama.jsonl")
    parser.add_argument("--output", type=str, default="tokenizer/tokenizer_32k.json")
    args = parser.parse_args()

    tokenizer = create_tokenizer(
        vocab_size=args.vocab_size,
        corpus_file=args.corpus,
        output_file=args.output,
    )
    test_tokenizer(tokenizer)
