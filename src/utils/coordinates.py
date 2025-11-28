from typing import Any, Tuple
from dataclasses import dataclass
import hashlib
import numpy as np


def get_hash_bucket(value: Any, buckets: int) -> int:
    """Calculates a deterministic hash bucket."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value) % buckets
    s = str(value).encode("utf-8")
    h = int.from_bytes(hashlib.sha256(s).digest()[:8], "big")
    return h % buckets


def get_range_bucket(value: Any, interval: int) -> int:
    """Calculates a range bucket (floor division)."""
    if value is None:
        return 0
    return int(value) // interval


@dataclass(frozen=True)
class NDimCoordinates:
    row_chunk: int
    col_chunk: int
    hash_coords: Tuple[int, ...]
    range_coords: Tuple[int, ...]
    version: int

    def to_filename(self) -> str:
        h_str = "-".join(map(str, self.hash_coords))
        r_str = "-".join(map(str, self.range_coords))
        return f"chunk_r{self.row_chunk}_c{self.col_chunk}_h{h_str}_r{r_str}_v{self.version}.parquet"
