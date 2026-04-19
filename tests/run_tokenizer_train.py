from cs336_basics.bpe import *
from tests.adapters import run_train_bpe
import yaml


def test_tiny_stories():
    vocab, merges = run_train_bpe(
            input_path='data/TinyStoriesV2-GPT4-train.txt',
            vocab_size=10_000,
            special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
    )
    save_tokenizer_yaml(vocab,merges,'tokenizer_TinyStories_valid.yaml')

def save_tokenizer_yaml(vocab, merges, output_path):
    vocab_serialized = {
        k: v.decode('utf-8', error="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }
    merges_serializable = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            {"vocab": vocab_serialized, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=True
        )

def load_tokenizer_yaml(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f);

    vocab = {
        int(k): v.encode('utf-8') if isinstance(v, str) else v
        for k, v in data['vocab'].items()
    }
    merges = [
        (a.encode('utf-8'), b.encode('utf-8'))
        for a, b in data['merges']
    ]

    return vocab, merges

if __name__ == "__main__":
    test_tiny_stories()