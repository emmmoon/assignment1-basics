from cs336_basics.bpe import *
from tests.adapters import run_train_bpe
import yaml


def test_tiny_stories():
    vocab, merges = run_train_bpe(
            input_path='data/TinyStoriesV2-GPT4-valid.txt',
            vocab_size=10_000,
            special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
    )
    

if __name__ == "__main__":
    test_tiny_stories()