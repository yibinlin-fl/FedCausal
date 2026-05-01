"""CSV logging utilities for Kaggle experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


class CSVLogger:
    """Append rows to a CSV file while writing the header once."""

    def __init__(
        self,
        path: str | Path,
        fieldnames: Iterable[str],
        reset: bool = False,
    ) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if reset or not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Mapping[str, object]) -> None:
        """Append one row."""
        clean_row = {field: row.get(field, "") for field in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)

    def log_many(self, rows: Iterable[Mapping[str, object]]) -> None:
        """Append multiple rows."""
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for row in rows:
                clean_row = {field: row.get(field, "") for field in self.fieldnames}
                writer.writerow(clean_row)
