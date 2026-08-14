import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from collections import Counter
import heapq
from dataclasses import dataclass
from tqdm import tqdm

# patch = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
patch = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass(frozen=True, slots=True)
class MaxPairEntry:
    count: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "MaxPairEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

class BPETrainerState:
    def __init__(self, special_tokens_bytes):
        self.word_counts = Counter()
        self.pair_counts = Counter()
        self.pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        self.pair_heap: list[MaxPairEntry] = []
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = {
                i: bytes([i])
                for i in range(256)
            }
        for stb in special_tokens_bytes:
            self.vocab[len(self.vocab)] = stb
        
    def _rebuild_pair_heap(self):
        self.pair_heap = [
            MaxPairEntry(count, pair)
            for pair, count in self.pair_counts.items()
            if count > 0
        ]
        heapq.heapify(self.pair_heap)
        
    def _push_pair_snapshot(self, pair: tuple[bytes, bytes]):
        count = self.pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(self.pair_heap, MaxPairEntry(count, pair))
        
    def initialize(self):
        for word, word_f in self.word_counts.items():
            pair_multiplicities = Counter(zip(word, word[1:]))
            for pair, occurrences in pair_multiplicities.items():
                self.pair_counts[pair] += (word_f * occurrences)
                self.pair_to_words.setdefault(pair, set()).add(word)
        self._rebuild_pair_heap()
                
    def add_vocab(self, word):
        self.vocab[len(self.vocab)] = word
                
    def best_pair(self):
        if len(self.pair_heap) > 4 * max(1, len(self.pair_counts)):
            self._rebuild_pair_heap()
        while self.pair_heap:
            entry = self.pair_heap[0]
            current_count = self.pair_counts.get(entry.pair, 0)
            if current_count == entry.count:
                return entry.pair
            heapq.heappop(self.pair_heap)
        return None
    
    def add_word_pairs(self, word, freq):
        pair_multiplicities = Counter(zip(word, word[1:]))
        for pair, occurrences in pair_multiplicities.items():
            self.pair_counts[pair] += (freq * occurrences)
            self.pair_to_words.setdefault(pair, set()).add(word)
            self._push_pair_snapshot(pair)

            
    def remove_word_pairs(self, word, freq):
        pair_multiplicities = Counter(zip(word, word[1:]))
        for pair, occurrences in pair_multiplicities.items():
            new_count = (self.pair_counts[pair] - freq * occurrences)
            if new_count <= 0:
                del self.pair_counts[pair]
            else:
                self.pair_counts[pair] = new_count
                self._push_pair_snapshot(pair)
            words = self.pair_to_words.get(pair)
            if words is not None:
                words.discard(word)
                if not words:
                    del self.pair_to_words[pair]
    
    def merge_word(self, word, pair):
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(word[i])
                i += 1
        return tuple(new_tokens)
        
    def merge_pair(self, pair):
        merged_bytes = pair[0] + pair[1]
        self.add_vocab(merged_bytes)
        self.merges.append(pair)
        affected_words = list(self.pair_to_words.get(pair, set()))
        for old_word in affected_words:
            if old_word not in self.word_counts:
                continue
            freq = self.word_counts[old_word]
            self.remove_word_pairs(old_word, freq)
            new_word = self.merge_word(old_word, pair)
            del self.word_counts[old_word]
            self.word_counts[new_word] += freq
            self.add_word_pairs(new_word, freq)

# special token split

def find_chunk_boundaries(
    file: BinaryIO,
    desired_chunk_size: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Needs future processing to handle special tokens.
    May return fewer chunks if the boundaries end up overlapping.
    """
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    desired_num_chunks = file_size // desired_chunk_size + 1
    chunk_boundaries = [i * desired_chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    mini_chunk_size = 4096
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) 
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_flag = False
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found_flag = True
                    break
            if found_flag:
                break
            initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# pretokenize

def scan_chunk(args) -> Counter[tuple[bytes, ...]]:

    input_path, start, end, special_pattern = args
    
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)
        chunk_text = chunk.decode("utf-8")
        if special_pattern == "":
            parts = [chunk_text]
        else:
            parts = re.split(special_pattern, chunk_text)
        
    counts = Counter()
    for part in parts:
        for match in re.finditer(patch, part):
            word = match.group()
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            counts[word_bytes] += 1
    return counts

def bpe_train_form_file(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE model from a file.
    Returns a tuple of (vocab, merges).
    """
    
    special_tokens_bytes = [st.encode("utf-8") for st in special_tokens]
    
    bpe_train_box = BPETrainerState(special_tokens_bytes)
    
    if vocab_size < len(bpe_train_box.vocab):
        raise ValueError("vocab_size must be at least initial vocabulary size")

    # get sliced bondaries
    with open(input_path, "rb") as f:
        # the chunk size func may be changed later
        desired_chunk_size = 16384*192
        chunk_boundary = find_chunk_boundaries(f, desired_chunk_size, special_tokens_bytes)
        
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    tasks = [(input_path, start, end, special_pattern) for start, end in zip(chunk_boundary[:-1], chunk_boundary[1:])]
    with Pool(processes=10) as pool:
        results = pool.imap_unordered(scan_chunk, tasks, chunksize=1)
        for counts in tqdm(results, total=len(tasks), desc="Pre-tokenizing", unit="chunk"):
            bpe_train_box.word_counts.update(counts)
    
    bpe_train_box.initialize()
        
    num_merges = vocab_size - len(bpe_train_box.vocab)

    with tqdm(total=num_merges, desc="BPE merging", unit="merge") as progress:
        while len(bpe_train_box.vocab) < vocab_size:
            target_pair = bpe_train_box.best_pair()
            if target_pair is None:
                break
            bpe_train_box.merge_pair(target_pair)
            progress.update(1)
        
    return (bpe_train_box.vocab, bpe_train_box.merges)
        
    