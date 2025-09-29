# factorization/lyrics.py
from typing import List
import numpy as np
from .base import Factorization


class LyricsFactorization(Factorization):
    """
    Factorize lyrics into interpretable components (lines).
    """

    def __init__(self, lyrics: str, by: str = "line"):
        """
        Args:
            lyrics: raw lyric string
            by: "line" (default) or "word"
        """
        super().__init__(lyrics)
        self.by = by

    def factorize(self) -> List[str]:
        if self.by == "line":
            self.components = [
                line.strip() for line in self.input_data.split("\n") if line.strip()
            ]
        elif self.by == "word":
            self.components = self.input_data.split()
        else:
            raise ValueError("Invalid factorization mode. Choose 'line' or 'word    '.")
        return self.components

    def reconstruct(self, mask: np.ndarray) -> str:
        if self.components is None:
            raise ValueError("Must call factorize() before reconstruct().")
        if len(mask) != len(self.components):
            raise ValueError("Mask length must equal number of components.")

        kept = [comp for keep, comp in zip(mask, self.components) if keep]

        if self.by == "line":
            return "\n".join(kept)
        else:  # word
            return " ".join(kept)
