from cs336_basics.bpe import bpe_train_form_file
import pickle
import time
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training bpe"
    )
    parser.add_argument("--input", type=Path, default=Path("data/data/TinyStoriesV2-GPT4-train.txt"),help="Input materials")
    parser.add_argument("--output", type=Path, default=Path("tests/_results/tinystories_bpe_10k.pkl"), help="Training result path")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size")
    return parser.parse_args()

def find_longest_words(vocab: dict[int, bytes]) -> list[bytes]:
    max_length = 0
    longest_words: list[bytes] = []
    for word in vocab.values():
        if len(word) > max_length:
            max_length = len(word)
            longest_words = [word]
        elif len(word) == max_length:
            longest_words.append(word)
    return longest_words
            
if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"文件不存在：{args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    vocab_size=args.vocab_size
    special_tokens=["<|endoftext|>"]
    start = time.perf_counter()
    vocab, merges = bpe_train_form_file(str(args.input), vocab_size, special_tokens)
    end = time.perf_counter()
    interval = end - start
    
    print("The length of vocabulary: ", len(vocab))
    print("Check special tokens:")
    for special_token in special_tokens:
        if special_token.encode("utf-8") in vocab.values():
            print("\t", special_token, "\t ✅")
        else:
            print("\t", special_token, "\t ❌")
    print("The number of merges: ", len(merges))
    print("-------------------------------------------------------")
    print("Use time:", interval)
    
    data = {
        "input_file": str(args.input),
        "vocab": vocab,
        "merges": merges,
        "special_tokens": special_tokens,
        "vocab_size": vocab_size,
        "train_seconds": interval
    }
    with args.output.open("wb") as f:
        pickle.dump(data, f)
    print("Save to: ",args.output)
    
    print("-------------------------------------------------------")
    longest_words = find_longest_words(vocab)
    print("Longest tokens:")

    for index, word in enumerate(longest_words, start=1):
        print(
            f"{index}. "
            f"text={word.decode('utf-8', errors='replace')!r}, "
            f"bytes={word!r}, "
            f"length={len(word)}"
        )