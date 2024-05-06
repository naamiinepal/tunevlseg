import gzip
import html
import itertools
import os
from collections.abc import Iterable
from functools import lru_cache
from typing import Final, TypeVar

import ftfy
import regex as re
import torch


@lru_cache(maxsize=1)
def default_bpe() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    cs = bs.copy()
    n = 0
    exp: Final = 2**8
    for b in range(exp):
        if b not in bs:
            bs.append(b)
            cs.append(exp + n)
            n += 1
    cs = map(chr, cs)
    return dict(zip(bs, cs, strict=True))


T = TypeVar("T")  # Declare type variable for generics


def get_pairs(word: Iterable[T]) -> set[tuple[T, T]]:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    return set(itertools.pairwise(word))


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer:
    def __init__(self, bpe_path: str = default_bpe()) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = self.get_merges(bpe_path)
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}

        vocab = self.get_vocab(merges)
        self.decoder = dict(enumerate(vocab))
        self.encoder = {v: k for k, v in self.decoder.items()}

        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    @staticmethod
    def get_vocab(merges: Iterable[Iterable[str]]) -> tuple[str, ...]:
        # Simplify the code above
        unicode_tokens = bytes_to_unicode().values()

        # add word boundary to unicode tokens
        word_boundaries = (t + "</w>" for t in unicode_tokens)

        # add special tokens
        special_tokens = ("<|startoftext|>", "<|endoftext|>")

        return (*unicode_tokens, *word_boundaries, *special_tokens)

    @staticmethod
    def get_merges(
        bpe_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> tuple[tuple[str, ...], ...]:
        with gzip.open(bpe_path, "rt", encoding="utf8") as f:
            merges = f.read().split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        return tuple(tuple(merge.split()) for merge in merges)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = (*token[:-1], token[-1] + "</w>")
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            _token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(_token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: Iterable[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        return (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )


_tokenizer = SimpleTokenizer()


def tokenize(
    texts: str | Iterable[str], context_length: int = 77, truncate: bool = False
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = (texts,)

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = ((sot_token, *_tokenizer.encode(text), eot_token) for text in texts)
    result: torch.LongTensor = torch.zeros(len(texts), context_length, dtype=torch.long)  # type:ignore

    for i, (text, tokens) in enumerate(zip(texts, all_tokens, strict=True)):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(
                    f"Input {text} is too long for context length {context_length}"
                )
            trunc_tokens = list(tokens[:context_length])
            trunc_tokens[-1] = eot_token
        else:
            trunc_tokens = tokens
        result[i, : len(trunc_tokens)] = torch.tensor(trunc_tokens)

    return result
