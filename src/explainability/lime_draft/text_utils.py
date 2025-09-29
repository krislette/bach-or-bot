import re
from typing import List, Sequence, Tuple, Optional, Union


class IndexedString:
    """
    Represents a string (lyrics) indexed by token.
    By default, tokens are split by line, but regex can be provided.
    """

    def __init__(
        self,
        text: str,
        bow: bool = True,
        split_expression: str = r"\n",
        mask_string: Optional[str] = None,
    ):
        self.text = text
        self.bow = bow
        self.split_expression = split_expression
        self.mask_string = mask_string or "..."

        # Tokenize
        self.tokens = [
            t for t in re.split(self.split_expression, text) if t.strip() != ""
        ]
        self.num_tokens = len(self.tokens)

    def num_words(self) -> int:
        """Return number of tokens (lines/words)."""
        return self.num_tokens

    def word(self, i: int) -> str:
        """Return token at index i."""
        return self.tokens[i]

    def inverse_removing(self, inactive: Sequence[int]) -> str:
        """
        Reconstruct text with tokens at indices in `inactive` removed/masked.
        """
        return "\n".join(
            self.mask_string if i in inactive else tok
            for i, tok in enumerate(self.tokens)
        )

    def __getitem__(self, i: int) -> str:
        return self.tokens[i]

    def __len__(self) -> int:
        return self.num_tokens

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self) -> str:
        return f"IndexedString(num_tokens={self.num_tokens})"


class IndexedCharacters:
    """
    Represents a string indexed by character.
    Used when char_level=True.
    """

    def __init__(self, text: str, mask_string: Optional[str] = None):
        self.text = text
        self.tokens = list(text)
        self.num_tokens = len(self.tokens)
        self.mask_string = mask_string or "…"

    def num_words(self) -> int:
        return self.num_tokens

    def word(self, i: int) -> str:
        return self.tokens[i]

    def inverse_removing(self, inactive: Sequence[int]) -> str:
        return "".join(
            self.mask_string if i in inactive else ch
            for i, ch in enumerate(self.tokens)
        )

    def __getitem__(self, i: int) -> str:
        return self.tokens[i]

    def __len__(self) -> int:
        return self.num_tokens

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self) -> str:
        return f"IndexedCharacters(num_tokens={self.num_tokens})"


class TextDomainMapper:
    """
    Maps explanation feature indices back to human-readable tokens.
    """

    def __init__(self, indexed: Union[IndexedString, IndexedCharacters]):
        self.indexed = indexed

    def map_exp_ids(self, exp: Sequence[Tuple[int, float]]) -> List[Tuple[str, float]]:
        """
        Convert explanation indices to (token, weight) pairs.
        """
        return [(self.indexed.word(idx), weight) for idx, weight in exp]

    def __repr__(self) -> str:
        return f"TextDomainMapper(type={type(self.indexed).__name__})"
