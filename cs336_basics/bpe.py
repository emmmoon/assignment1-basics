import json
import os
import gc
import multiprocessing as mp
from collections import Counter, defaultdict
import regex as re
from cs336_basics import c_bpe

N_BYTES = 256
CHUNK_SIZE = 1024 * 1024 * 10  # 50MB chunks

def _worker_pretokenize_and_count(args):
    chunk_bytes, special_pattern_str, normal_pattern_str = args

    special_pattern = re.compile(special_pattern_str)
    normal_pattern = re.compile(normal_pattern_str)
    chunk = chunk_bytes.decode('utf-8')

    local_counts = Counter()
    blocks = special_pattern.split(chunk)
    for block in blocks:
        if not block:
            continue
        for match in normal_pattern.finditer(block):
            local_counts[match.group(0)] += 1
    
    return local_counts


class BPE_Trainer:
    @staticmethod
    def _chunk_document_streaming(
        input_path: str | os.PathLike,
        chunk_size: int = CHUNK_SIZE,
        special_token: str = "<|endoftext|>",
    ):
        """
        Reads 'path' in streaming fashion, yielding chunks of text that
        each end on a '<|endoftext|>' boundary.
        """
        leftover = b""
        special_token_bytes = special_token.encode('utf-8')
        token_bytes_len = len(special_token_bytes)
        with open(input_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                block = leftover + chunk
                leftover = b""
                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token_bytes)

                if last_eot_idx == -1:
                    leftover = block
                else:
                    yield block[: last_eot_idx + token_bytes_len]
                    leftover = block[last_eot_idx + token_bytes_len :]
        
        # yield leftover text
        if leftover:
            yield leftover

    @staticmethod
    def _count_pairs(
        word_counts: dict[str, int],
        word_encoding: dict[str, list[int]],
        pair_string: dict[tuple[int, int], bytes],
        pair_to_words: dict[tuple[int, int], set[str]],
        vocabulary: dict[int, bytes]
    ) -> dict[tuple[bytes, bytes], int]:
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encoding[word]
            for i in range(len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_string:
                    pair_string[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])
                pair_to_words[pair].add(word)

        return pair_counts
    
    @staticmethod
    def _update_pair_counts(
        max_pair: tuple[int, int],
        word_counts: dict[str, int],
        affected_words: set[str],
        pair_counts: dict[tuple[int, int], int],
        word_encoding: dict[str, list[int]],
        pair_string: dict[tuple[int, int], bytes],
        vocabulary: dict[int, bytes],
        pair_to_words: dict[tuple[int, int], set[str]],
        new_id: int,
    ):
        for word in list(affected_words):
            word_tokens = word_encoding[word]
            count = word_counts[word]

            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                    del pair_string[old_pair]
                else:
                    pair_to_words[old_pair].discard(word)
            
            i = 0
            new_tokens = []

            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == max_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
        
            word_encoding[word] = new_tokens

            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[new_pair] += count
                pair_to_words[new_pair].add(word)
                if new_pair not in pair_string:
                    pair_string[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])

        
    @staticmethod
    def _merge_a_pair(
        word_counts: dict[str, int],
        pair_counts: dict[tuple[int, int], int],
        word_encoding: dict[str, list[int]],
        pair_string: dict[tuple[int, int], bytes],
        vocabulary: dict[int, bytes],
        size: int,
        pair_to_words: dict[tuple[int, int], set[str]],
        merges: list[tuple[bytes, bytes]],
    ) -> None:
            max_pair, max_count = max(pair_counts.items(), key=lambda x: (x[1], pair_string[x[0]]))
            merge_bytes = vocabulary[max_pair[0]] + vocabulary[max_pair[1]]
            vocabulary[size] = merge_bytes
            new_id = size

            affected_words = pair_to_words[max_pair]
            BPE_Trainer._update_pair_counts(
                max_pair,
                word_counts,
                affected_words, 
                pair_counts, 
                word_encoding, 
                pair_string, 
                vocabulary,
                pair_to_words,
                new_id,
            )
        
            merges.append((vocabulary[max_pair[0]], vocabulary[max_pair[1]]))  


    def _pretokenize_and_count(self,
        input_path: str | os.PathLike,
        special_tokens: list[str],
        num_workers: int = None,
    ) -> dict[str, int]:
        if num_workers is None:
            num_workers = int(max(1, mp.cpu_count() / 2))

        # pre-compile regex
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # build split pattern for special tokens
        special_pattern = "|".join(re.escape(token) for token in special_tokens)

        def chunk_arg_generator():
            for chunk in self._chunk_document_streaming(input_path):
                yield chunk, special_pattern, pattern

        final_counts = Counter()

        with mp.Pool(num_workers) as pool:
            for local_counts in pool.imap_unordered(_worker_pretokenize_and_count, chunk_arg_generator(), chunksize=1):
                final_counts.update(local_counts)

        return dict(final_counts)


    def train(self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **args,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
        word_counts = self._pretokenize_and_count(input_path, special_tokens)
        vocabulary = {i: bytes([i]) for i in range(N_BYTES)}
        for i, token in enumerate(special_tokens, start=N_BYTES):
            vocabulary[i] = token.encode('utf-8')
        cpp_vocab = {i: list(v) for i, v in vocabulary.items()}
        size = N_BYTES + len(special_tokens)
        merges = []

        word_to_ids = {word: idx for idx, word in enumerate(word_counts.keys())}
        ids_to_words = {idx: word for word, idx in word_to_ids.items()}
        wordids_counts = {word_to_ids[word]: count for word, count in word_counts.items()}

        word_encoding = {}
        for word in word_counts:
            word_encoding[word] = list(word.encode('utf-8'))

        wordid_encoding = {word_to_ids[word]: encoding for word, encoding in word_encoding.items()}

        del word_counts
        del word_to_ids
        del ids_to_words
        # 视情况也可以把其他不需要的中间变量 del 掉
        gc.collect() # 强制回收内存

        vocab, merges = c_bpe.train_bpe_cpp(
            wordids_counts,
            wordid_encoding,
            cpp_vocab,
            size,
            vocab_size
        )

        vocabulary = {k: bytes(v) for k, v in vocab.items()}
        fin_merges = [
            (vocabulary[a_id], vocabulary[b_id]) 
            for a_id, b_id in merges
        ]
            
        
        # pair_string = {}
        # pair_to_words = defaultdict(set)
        # pair_counts = self._count_pairs(word_counts, 
        #     word_encoding, 
        #     pair_string, 
        #     pair_to_words,
        #     vocabulary
        # )

        # while size < vocab_size:
        #     BPE_Trainer._merge_a_pair(word_counts,
        #         pair_counts,
        #         word_encoding,
        #         pair_string,
        #         vocabulary,
        #         size,
        #         pair_to_words,
        #         merges
        #     )
        #     size += 1
        #     if size % 100 == 0:
        #         print(f"Currently at vocab size: {size} / {vocab_size}")

        return vocabulary, fin_merges
    

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [
            token.encode('utf-8') for token in self.special_tokens
        ]
        self.vocab_to_ids = {v: k for k, v in vocab.items()}

        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_ids:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_ids[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            vocab = json.load(vf)
            vocab = {int(k): bytes(v, 'latin-1') if isinstance(v, str) else bytes(v)
                    for k, v in vocab.items()}
        
        with open(vocab_path, 'r', encoding='utf-8') as mf:
            lines = mf.readlines()
            merges_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merges_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

        
