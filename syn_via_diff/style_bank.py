"""Prompt management utilities for style diversification."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator, List, Optional

import yaml


class StyleBank:
    """Cycles through a collection of positive prompts."""

    def __init__(self, base_prompt: str, variations: Optional[List[str]] = None):
        self.base_prompt = base_prompt
        self.variations = variations or []
        self._cycle: Iterator[str] | None = None

    def next_prompt(self, step: int) -> str:
        if not self.variations:
            return self.base_prompt
        if self._cycle is None:
            self._cycle = itertools.cycle(self.variations)
        return next(self._cycle)

    @classmethod
    def from_yaml(cls, path: Path, base_prompt: str) -> "StyleBank":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        variations = data.get("prompts", []) if isinstance(data, dict) else data
        if not isinstance(variations, list):
            raise ValueError("Prompt bank YAML must define a list under the 'prompts' key or be a list.")
        return cls(base_prompt=base_prompt, variations=list(map(str, variations)))
