from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC, Sequence, Strip
import os

# Constants
NUM_SENTINELS = 100
VOCAB_SIZE = 8192
CORPUS_FILE = "data/Pretrain/slimpajama.jsonl"
MODEL_PREFIX = "tokenizer/tokenizer"


def create_tokenizer():
    # 1) Initialize a BPE tokenizer with an unk token
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2) Normalization: NFKC + strip
    tokenizer.normalizer = Sequence([NFKC(), Strip()])

    # 3) Pre-tokenization: byte-level (whitespace+punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    special_tokens = ["<unk>", "<s>", "</s>",
        "<user>", "<assistant>", "<pad>"
    ]

    # 5) Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True
    )

    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_FILE}")

    tokenizer.train([CORPUS_FILE], trainer)


    # 7) Decoder (optional)
    tokenizer.decoder = decoders.ByteLevel()
    
    tokenizer.save(f"{MODEL_PREFIX}.json")
    return tokenizer


def test_tokenizer(tokenizer):
    """Test the tokenizer with various inputs"""
    print("\nTesting special token encoding:")
    test_inputs = ["<user>", "<assistant>", "Hello, world!"]
    for text in test_inputs:
        output = tokenizer.encode(text)
        print(f"{text} -> ids={output.ids}, tokens={output.tokens}")

if __name__ == "__main__":
    tokenizer = create_tokenizer()
    test_tokenizer(tokenizer)
