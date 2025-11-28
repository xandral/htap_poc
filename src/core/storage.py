from pathlib import Path
import threading
import time
import json
from typing import Dict, List, Tuple, Any, Set, Union, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np

from src.core.catalog import FileCatalog
from src.utils.sequence_manager import SequenceManager
from src.utils.coordinates import get_hash_bucket, get_range_bucket
from src.core.filters import FilterEngine


# --- SCHEMA MANAGER ---
class SchemaManager:
    """
    Manages the mapping between Column Names and their global Index.
    Ensures deterministic vertical pruning even if file names change.
    """

    def __init__(self, base_path: Path):
        self.path = base_path / "schema.json"
        self.lock = threading.Lock()
        self.column_map: Dict[str, int] = {}  # col_name -> index
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                self.column_map = json.load(f)

    def update(self, column_names: List[str]):
        """Updates schema if new columns are seen (Append-only)."""
        with self.lock:
            changed = False
            # Filter out global_id, we treat it separately
            data_cols = [c for c in column_names if c != "global_id"]

            # Simple append logic: if col unknown, append to end
            current_max_idx = max(self.column_map.values()) if self.column_map else -1

            for col in data_cols:
                if col not in self.column_map:
                    current_max_idx += 1
                    self.column_map[col] = current_max_idx
                    changed = True

            if changed:
                with open(self.path, "w") as f:
                    json.dump(self.column_map, f)

    def get_chunk_index(self, col_name: str, chunk_size: int) -> Optional[int]:
        """Mathematically calculates the Vertical Chunk Index."""
        if col_name == "global_id":
            return None  # Present in all chunks
        if col_name not in self.column_map:
            return None  # Unknown col

        col_idx = self.column_map[col_name]
        return col_idx // chunk_size


# --- CORE STORAGE ---
class NDimStorage:
    def __init__(
        self,
        base_path: str,
        chunk_rows=100_000,
        chunk_cols=32,
        hash_dims=None,
        range_dims=None,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.chunk_rows = chunk_rows
        self.chunk_cols = chunk_cols
        self.hash_dims = hash_dims or {}
        self.range_dims = range_dims or {}
        self.hash_keys = list(self.hash_dims.keys())
        self.range_keys = list(self.range_dims.keys())

        self.catalog = FileCatalog(self.base_path)
        self.sequence = SequenceManager(self.base_path)
        self.schema = SchemaManager(self.base_path)
        self.io_pool = ThreadPoolExecutor(max_workers=8)

    # --- INGESTION (VECTORIZED) ---
    def write_batch(self, table: pa.Table):
        t0 = time.time()
        self.schema.update(table.column_names)

        # 1. Global ID
        if "global_id" not in table.column_names:
            start_id = self.sequence.allocate_ids(len(table))
            ids = np.arange(start_id, start_id + len(table))
            table = table.append_column("global_id", pa.array(ids))
        else:
            ids = table["global_id"].to_numpy()

        # 2. Vectorized Calc
        row_chunks = ids // self.chunk_rows

        hash_cols = []
        for k in self.hash_keys:
            data = table[k].to_numpy()
            if np.issubdtype(data.dtype, np.integer):
                hash_cols.append(data % self.hash_dims[k])
            else:
                hash_cols.append(
                    np.vectorize(
                        lambda x: get_hash_bucket(x, self.hash_dims[k]), otypes=[int]
                    )(data)
                )

        range_cols = []
        for k in self.range_keys:
            range_cols.append(table[k].to_numpy() // self.range_dims[k])

        # 3. Grouping
        partitions = defaultdict(list)
        h_iter = zip(*hash_cols) if hash_cols else [()] * len(table)
        r_iter = zip(*range_cols) if range_cols else [()] * len(table)

        for i, (r_c, h, r) in enumerate(zip(row_chunks, h_iter, r_iter)):
            partitions[(r_c, h, r)].append(i)

        # 4. Write
        futures = []
        for (r_chk, h_c, r_c), indices in partitions.items():
            sub = table.take(indices)
            data_cols = [c for c in sub.column_names if c != "global_id"]

            for c_idx, i in enumerate(range(0, len(data_cols), self.chunk_cols)):
                cols = data_cols[i : i + self.chunk_cols]
                chunk_table = sub.select(["global_id"] + cols)

                h_str = "-".join(map(str, h_c))
                r_str = "-".join(map(str, r_c))
                logical_key = f"chunk_r{r_chk}_c{c_idx}_h{h_str}_r{r_str}"
                ver = self.catalog.get_next_version(logical_key)

                futures.append(
                    self.io_pool.submit(
                        pq.write_table,
                        chunk_table,
                        self.base_path / f"{logical_key}_v{ver}.parquet",
                        compression="zstd",
                    )
                )

        for f in as_completed(futures):
            f.result()
        print(f"✓ Ingested {len(table)} rows in {time.time()-t0:.2f}s")

    # --- PRUNING (METADATA + SCHEMA) ---
    def _evaluate_pruning(self, expression, candidates):
        if not expression:
            return candidates
        if isinstance(expression, tuple):
            return self._filter_leaf(expression, candidates)
        if isinstance(expression, list) and len(expression) == 1:
            return self._evaluate_pruning(expression[0], candidates)

        curr, op = None, "AND"
        for item in expression:
            if isinstance(item, str):
                if item.upper() in ["AND", "OR"]:
                    op = item.upper()
                continue
            subset = self._evaluate_pruning(item, candidates)
            if curr is None:
                curr = subset
            else:
                curr = curr.intersection(subset) if op == "AND" else curr.union(subset)
        return curr if curr is not None else candidates

    def _filter_leaf(self, cond, candidates):
        col, op, val = cond
        if op != "=":
            return candidates

        tr, th, trg = None, {}, {}
        if col == "global_id":
            tr = val // self.chunk_rows
        if col in self.hash_dims:
            th[self.hash_keys.index(col)] = get_hash_bucket(val, self.hash_dims[col])
        if col in self.range_dims:
            trg[self.range_keys.index(col)] = get_range_bucket(
                val, self.range_dims[col]
            )

        if tr is None and not th and not trg:
            return candidates

        filtered = set()
        for p in candidates:
            parts = p.name.split("_")
            if tr is not None and tr != int(parts[1][1:]):
                continue

            h_vals = [int(x) for x in parts[3][1:].split("-")]
            if any(h_vals[i] != b for i, b in th.items()):
                continue

            r_vals = [int(x) for x in parts[4][1:].split("-")]
            if any(r_vals[i] != b for i, b in trg.items()):
                continue

            filtered.add(p)
        return filtered

    def _get_candidates(self, filters, columns=None):
        candidates = self.catalog.get_active_files()
        if filters:
            candidates = self._evaluate_pruning(filters, set(candidates))

        # Rectangular Pruning (Vertical)
        if columns:
            target_chunks = set()
            for col in columns:
                idx = self.schema.get_chunk_index(col, self.chunk_cols)
                if idx is None:
                    return list(candidates)  # Unknown col, scan all
                target_chunks.add(idx)

            final = []
            for p in candidates:
                try:
                    if int(p.name.split("_")[2][1:]) in target_chunks:
                        final.append(p)
                except:
                    final.append(p)
            return final

        return list(candidates)

    # --- SCAN ---
    def scan(self, filters, columns=None):
        candidates = self._get_candidates(filters, columns)
        if not candidates:
            return pa.Table.from_pylist([])

        needed = set(["global_id"])
        if columns:
            needed.update(columns)
        needed.update(FilterEngine.extract_columns(filters))
        needed_list = list(needed)

        results = []

        def process(path):
            try:
                schema = pq.read_schema(path)
                avail = [c for c in needed_list if c in schema.names]
                if not avail:
                    return None
                # Smart Skip: if only ID matches but we need logic/data, skip
                if len(avail) == 1 and avail[0] == "global_id" and len(needed_list) > 1:
                    return None

                t = pq.read_table(path, columns=avail)
                if filters:
                    mask = FilterEngine.evaluate(t, filters)
                    if mask is None or pc.sum(mask.cast("int8")).as_py() == 0:
                        return None
                    t = t.filter(mask)
                return t
            except:
                return None

        futures = [self.io_pool.submit(process, p) for p in candidates]
        for f in as_completed(futures):
            if res := f.result():
                results.append(res)

        if not results:
            return pa.Table.from_pylist([])
        full = pa.concat_tables(results)
        if columns:
            full = full.select([c for c in columns if c in full.column_names])
        return full

    # --- UPDATE (2-PHASE) ---
    def update(self, filters, updates, unsafe=False):
        if not filters and not unsafe:
            raise ValueError("Safety Error: Empty filters")
        t0 = time.time()

        # Phase 1: Identify IDs
        print("   Phase 1: Scanning for IDs...")
        target_table = self.scan(filters, columns=[])
        if not target_table or len(target_table) == 0:
            return
        target_ids = target_table["global_id"].to_numpy()

        # Phase 2: Write specific chunks
        cols_to_upd = list(updates.keys())
        candidates = self._get_candidates(
            filters, cols_to_upd
        )  # Narrow candidates vertically

        files_upd = 0

        def process_upd(path):
            try:
                t = pq.read_table(path)
                # Filter by ID match (Join)
                mask_np = np.isin(t["global_id"].to_numpy(), target_ids)
                if not np.any(mask_np):
                    return 0

                mask = pa.array(mask_np)
                new_cols = []
                dirty = False
                for col in t.column_names:
                    if col in updates:
                        new_cols.append(
                            pc.if_else(
                                mask, pa.scalar(updates[col], type=t[col].type), t[col]
                            )
                        )
                        dirty = True
                    else:
                        new_cols.append(t[col])

                if dirty:
                    nt = pa.Table.from_arrays(new_cols, names=t.column_names)
                    lk = path.name[: path.name.rfind("_v")]
                    ver = self.catalog.get_next_version(lk)
                    pq.write_table(
                        nt, self.base_path / f"{lk}_v{ver}.parquet", compression="zstd"
                    )
                    return 1
                return 0
            except:
                return 0

        futures = [self.io_pool.submit(process_upd, p) for p in candidates]
        for f in as_completed(futures):
            files_upd += f.result()

        self.catalog.refresh()
        print(
            f"Update: {len(target_ids)} rows in {files_upd} chunks ({time.time()-t0:.2f}s)"
        )

    def vacuum(self):
        with self.catalog.lock:
            active = self.catalog.active_versions.copy()
        deleted = 0
        for f in self.base_path.iterdir():
            if not f.name.endswith(".parquet"):
                continue
            try:
                lk = f.name[: f.name.rfind("_v")]
                ver = int(f.name[f.name.rfind("_v") + 2 : -8])
                if ver < active.get(lk, 0):
                    f.unlink()
                    deleted += 1
            except:
                continue
        print(f"Vacuum cleaned {deleted} files.")

    def close(self):
        self.io_pool.shutdown()
