import re
import numpy as np
from lime.lime_text import IndexedString


class LineIndexedString(IndexedString):
    def __init__(self, raw_string, bow=True, mask_string=None):
        """
        Initialize line-based text indexing for lyrics perturbation in MusicLIME.

        Parameters
        ----------
        raw_string : str
            Raw lyrics text to be processed
        bow : bool, default=True
            Bag-of-words flag (maintained for LIME compatibility)
        mask_string : str, optional
            String to use for masking removed lines
        """
        self.raw = raw_string
        self.mask_string = mask_string
        self.bow = bow

        # Split by lines instead of words
        self.as_list = self._split_by_lines(raw_string)
        self.as_np = np.array(self.as_list)

        # Create word positions mapping (for compatibility)
        self.positions = list(range(len(self.as_list)))
        self.string_start = [0] * len(self.as_list)

    def _split_by_lines(self, text):
        """
        Split lyrics text into meaningful lines, filtering out metadata.

        Parameters
        ----------
        text : str
            Raw lyrics text with potential metadata

        Returns
        -------
        list of str
            Processed lyrics lines with metadata removed
        """
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            # Skip metadata lines
            if not line or re.match(r"^\[.*\]$", line) or re.match(r"^\(.*\)$", line):
                continue
            processed_lines.append(line)

        return processed_lines

    def inverse_removing(self, words_to_remove):
        """
        Reconstruct lyrics text by removing specified line indices.

        Parameters
        ----------
        words_to_remove : array-like
            Indices of lyrics lines to remove from reconstruction

        Returns
        -------
        str
            Reconstructed lyrics text with specified lines removed
        """
        # Keep lines not in words_to_remove
        kept_lines = [
            self.as_list[i]
            for i in range(len(self.as_list))
            if i not in words_to_remove
        ]
        return "\n".join(kept_lines)

    def num_words(self):
        """
        Get total number of lyrics lines (called 'words' for LIME compatibility).

        Returns
        -------
        int
            Number of lyrics lines available for perturbation
        """
        return len(self.as_list)

    def word(self, id_):
        """
        Get lyrics line content by index.

        Parameters
        ----------
        id_ : int
            Index of the lyrics line to retrieve

        Returns
        -------
        str
            Content of the specified lyrics line
        """
        return self.as_list[id_]
