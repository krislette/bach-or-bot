"""Lyrics factorization module for MusicLIME explainability."""

from typing import List, Optional
from .base import Factorization


class LyricsFactorization(Factorization):
    """Factorization for lyrics text (line-by-line segmentation)."""

    def __init__(
        self,
        lyrics_lines: List[str],
        composition_fn: Optional[callable] = None,
    ):
        self.lyrics_lines = [line.strip() for line in lyrics_lines if line.strip()]
        self._components_names = [
            f"lyrics_line_{i}" for i in range(len(self.lyrics_lines))
        ]

        if composition_fn is None:
            composition_fn = self._default_lyrics_composition
        self._composition_fn = composition_fn

    def _default_lyrics_composition(self, selected_lines: List[str]) -> str:
        """Compose selected lyrics lines into a single string."""
        return "\n".join(line for line in selected_lines if line.strip())

    def retrieve_components(
        self, selection_order: Optional[List[int]] = None
    ) -> List[str]:
        """Retrieve selected lyrics lines."""
        if selection_order is None:
            selection_order = list(range(len(self.lyrics_lines)))

        selected_lines = []
        for idx in selection_order:
            if 0 <= idx < len(self.lyrics_lines):
                selected_lines.append(self.lyrics_lines[idx])
            else:
                selected_lines.append("")  # Empty string for invalid indices

        return selected_lines

    def compose_model_input(self, components: Optional[List[int]] = None) -> str:
        """Compose lyrics from selected components."""
        selected_lines = self.retrieve_components(components)
        return self._composition_fn(selected_lines)

    def get_number_components(self) -> int:
        return len(self.lyrics_lines)

    def get_ordered_component_names(self) -> List[str]:
        return self._components_names
