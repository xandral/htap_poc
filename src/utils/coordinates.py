from typing import Any, Tuple
from dataclasses import dataclass
import hashlib
import numpy as np
import json


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


# deprecated
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
        return f"chunk_r{self.row_chunk}_c{self.col_chunk}_h{h_str}_rg{r_str}_v{self.version}.parquet"


class NumpyEncoder(json.JSONEncoder):
    """
    Gestisce la serializzazione dei tipi NumPy (float32, int64, ndarray)
    per il modulo JSON standard, convertendoli nei tipi Python nativi.
    """

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            # Converti i float NumPy in float standard di Python
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            # Converti gli interi NumPy in int standard di Python
            return int(obj)
        if isinstance(obj, np.ndarray):
            # Converti gli array NumPy in liste standard di Python
            return obj.tolist()
        # Se non è un tipo gestito, usa il comportamento di default
        return json.JSONEncoder.default(self, obj)
