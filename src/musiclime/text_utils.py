import re
import numpy as np
from lime.lime_text import IndexedString


class LineIndexedString(IndexedString):
    def __init__(self, raw_string, bow=True, mask_string=None):
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
        # Keep lines not in words_to_remove
        kept_lines = [
            self.as_list[i]
            for i in range(len(self.as_list))
            if i not in words_to_remove
        ]
        return "\n".join(kept_lines)

    def num_words(self):
        return len(self.as_list)

    def word(self, id_):
        return self.as_list[id_]
