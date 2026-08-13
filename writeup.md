# Assignment 1

## Problem (unicode1):  Understanding Unicode (1 point)

**Q: What Unicode character does chr(0) return?**

A: '\x00'

**Q: How does this character’s string representation (__repr__()) differ from its printed representation?**

A: repr(chr(0)) returns '\x00', using the hexadecimal escape sequence \x00 to make the invisible NUL character visible.
When printed, the NUL character itself is sent to the output, but it normally has no visible glyph, so it appears blank. (print() also adds a newline by default.)

**Q: What happens when this character occurs in text? It may be helpful to play around with the 
following in your Python interpreter and see if it matches your expectations:**
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

The interactive interpreter uses `repr()`, which displays invisible characters such as NUL as `\x00`. `print()` writes the actual character to the terminal, but NUL has no visible representation, so it appears absent.

## Problem (unicode2):  Unicode Encodings (3 points)

**What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than 
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various 
input strings.**

UTF-8 为 1–4 字节，UTF-16 为 2/4 字节，UTF-32 为 4 字节；在ASCII/拉丁字符占比较高时的材料中， UTF-8 更紧凑。更短的字节序列意味着更少的存储、I/O 和训练开销，也减少 tokenizer 学习冗余编码模式所需的 merges。UTF-8 与 ASCII 兼容，不涉及字节序或 BOM，生态支持也最广泛。

**Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results**
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

encode成多个字节时，会有"10x..." "110xxxxx"等标号来表明字节是否结束/是否为开头；多个字节通常会被一起解码，且解码时会对对应内容进行检查。如果将多个字节拆分开逐个解码，则解码函数找不到对应的后续/开始字节，就会报错。
```python
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
> Tip： 可以查看/检查函数encode和decode的具体实现过程并且猜测其为什么出错。

**Give a two-byte sequence that does not decode to any Unicode character(s).**

```
"10111111 10111111"
```
起始字节不以"110..."开头，则非合法utf-8字节。

## Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points)

**(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM**

Memory：833 MiB
Time：143s
longest token: ' accomplishment'/' disappointment'/' responsibility'

**(b) Profile your code. What part of the tokenizer training process takes the most time?**

pretokenize.