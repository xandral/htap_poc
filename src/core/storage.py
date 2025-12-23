from pathlib import Path
import threading
import time
import json
import os
from typing import Dict, List, Set, Optional, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.schema_manager import ChunkMetadataManager

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np

from src.core.vector_operations import VectorOpSerializer
from src.utils.sequence_manager import SequenceManager, FileCatalog
from src.utils.coordinates import get_hash_bucket, get_range_bucket, NumpyEncoder
from src.utils.filters import FilterEngine

from src.utils.schema_manager import SchemaManager
from src.utils.coordinates import get_hash_bucket



class NDimStorage:
    """
    Multi-dimensional storage engine with merge-on-read updates.
    """

    def __init__(
        self,
        base_path: str,
        chunk_rows=100_000,
        chunk_cols=32,
        hash_dims=None,
        range_dims=None,
        global_id=None,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)

        # Configuration
        self.chunk_rows = chunk_rows
        self.chunk_cols = chunk_cols
        self.hash_dims = hash_dims or {}
        self.range_dims = range_dims or {}
        self.hash_keys = list(self.hash_dims.keys())
        self.range_keys = list(self.range_dims.keys())
        self.global_id = global_id

        # Managers
        self.catalog = FileCatalog(self.base_path)
        self.sequence = SequenceManager(self.base_path)
        self.schema = SchemaManager(self.base_path)
        self.io_pool = ThreadPoolExecutor(max_workers=1)
        self.chunk_metadata = ChunkMetadataManager(self.base_path)

        # Cache: {logical_key: {tid: table}}
        self._in_memory_updates: Dict[str, Dict[int, pa.Table]] = {}
        self._cache_lock = threading.Lock()

        # Patch log: {tid: {logical_key: patch_metadata}}
        self.patch_log: Dict[int, Dict[str, Dict]] = {}
        self.active_tids = {}  # tid: timestamp

        # Load existing configuration
        self._load_dim_configs()

    # --- Configuration Management ---

    def _load_dim_configs(self):
        """Load hash/range dimensions from persisted metadata."""
        for col_name, meta in self.schema.metadata.items():
            if meta.get("hash_buckets"):
                if col_name not in self.hash_dims:
                    self.hash_dims[col_name] = meta["hash_buckets"]
                    if col_name not in self.hash_keys:
                        self.hash_keys.append(col_name)

            if meta.get("range_interval"):
                if col_name not in self.range_dims:
                    self.range_dims[col_name] = meta["range_interval"]
                    if col_name not in self.range_keys:
                        self.range_keys.append(col_name)

    def _save_dim_configs(self):
        """Persist hash/range configurations into metadata."""
        with self.schema.lock:
            for col in self.hash_keys:
                if col in self.schema.metadata:
                    self.schema.metadata[col]["hash_buckets"] = self.hash_dims[col]
                else:
                    self.schema.metadata[col] = {"hash_buckets": self.hash_dims[col]}

            for col in self.range_keys:
                if col in self.schema.metadata:
                    self.schema.metadata[col]["range_interval"] = self.range_dims[col]
                else:
                    self.schema.metadata[col] = {"range_interval": self.range_dims[col]}

            self.schema._save()

    # --- Write Path ---
    def write_batch(self, table: pa.Table):
        t0 = time.time()
        tid = self.sequence.allocate_tids()
        self.active_tids[tid] = t0
        
        # Update schema 
        self.schema.update(table.column_names, global_id=self.global_id, table=table)
        self._save_dim_configs()

        # 1. Global ID 
        if self.global_id is None or self.global_id not in table.column_names:
            start_id = self.sequence.allocate_ids(len(table))
            ids = np.arange(start_id, start_id + len(table))
            table = table.append_column("global_id", pa.array(ids))
        else:
            ids = table[self.global_id].to_numpy()

        # 2. Partitioning 
        partitions = self._compute_partitions(table, ids)

        # 3. Write 
        futures = []
        for partition_key, indices in partitions.items():
            sub_table = table.take(indices)
            futures.extend(self._write_partition(partition_key, sub_table))

        # 4. Wait 
        for f in as_completed(futures):
            f.result()
        
        del self.active_tids[tid]
        print(f" Ingested {len(table)} rows in {time.time()-t0:.2f}s")
    
    def _compute_partitions(self, table: pa.Table, ids: np.ndarray) -> dict:
        
        row_chunks = ids // self.chunk_rows
        
        # Hash dimensions 
        hash_cols = []
        for k in self.hash_keys:
            if k not in table.column_names:
                continue
            data = table[k].to_numpy()
            if np.issubdtype(data.dtype, np.integer):
                hash_cols.append(data % self.hash_dims[k])
            else:
                hash_cols.append(
                    np.vectorize(
                        lambda x: get_hash_bucket(x, self.hash_dims[k]), 
                        otypes=[int]
                    )(data)
                )
        
        # Range dimensions
        range_cols = []
        for k in self.range_keys:
            if k not in table.column_names:
                continue
            range_cols.append(table[k].to_numpy() // self.range_dims[k])
        
        # Grouping
        partitions = defaultdict(list)
        h_iter = zip(*hash_cols) if hash_cols else [()] * len(table)
        r_iter = zip(*range_cols) if range_cols else [()] * len(table)
        
        for i, (r_c, h, r) in enumerate(zip(row_chunks, h_iter, r_iter)):
            partitions[(r_c, h, r)].append(i)
        
        return partitions

    def _write_partition(self, partition_key: tuple, sub_table: pa.Table) -> list:
        r_chk, h_c, r_c = partition_key
        futures = []
        
        # Data columns
        data_cols = [c for c in sub_table.column_names 
                    if c != self.schema.column_map["global_id"]]
        
        # Vertical chunking
        for c_idx, i in enumerate(range(0, len(data_cols), self.chunk_cols)):
            cols = data_cols[i : i + self.chunk_cols]
            chunk_table = sub_table.select([self.schema.column_map["global_id"]] + cols)
            
            # Logical key
            h_str = "-".join(map(str, h_c))
            r_str = "-".join(map(str, r_c))
            logical_key = f"chunk_r{r_chk}_c{c_idx}_h{h_str}_r{r_str}"
            
            # Smart append logic
            current_rows = self.chunk_metadata.get_row_count(logical_key)
            available = self.chunk_rows - current_rows
            new_rows = len(chunk_table)
            
            if available == 0 or current_rows == 0:
                # New chunk
                ver = self.catalog.get_next_version(logical_key)
                futures.append(
                    self.io_pool.submit(
                        self._write_file, logical_key, ver, chunk_table, new_rows
                    )
                )
            elif new_rows <= available:
                # Fits in existing chunk
                futures.append(
                    self.io_pool.submit(
                        self._append_file, logical_key, chunk_table, current_rows + new_rows
                    )
                )
            else:
                # Split: fill existing + create new
                fill_part = chunk_table.slice(0, available)
                futures.append(
                    self.io_pool.submit(
                        self._append_file, logical_key, fill_part, self.chunk_rows
                    )
                )
                
                # Overflow handling 
                overflow = chunk_table.slice(available)
                remaining = len(overflow)
                offset = 0
                
                while remaining > 0:
                    size = min(remaining, self.chunk_rows)
                    part = overflow.slice(offset, size)
                    ver = self.catalog.get_next_version(logical_key)
                    futures.append(
                        self.io_pool.submit(
                            self._write_file, logical_key, ver, part, size
                        )
                    )
                    remaining -= size
                    offset += size
        
        return futures
        
    def _write_file(self, logical_key: str, version: int, data: pa.Table, row_count: int):
        output_path = self.base_path / f"{logical_key}_v{version}.parquet"
        pq.write_table(data, output_path, compression="zstd")
        self.chunk_metadata.update_row_count(logical_key, row_count)
    
    def _append_file(self, logical_key: str, new_data: pa.Table, new_total: int):

        # Find latest version
        versions = []
        for f in self.base_path.glob(f"{logical_key}_v*.parquet"):
            try:
                ver_str = f.stem.split('_v')[-1]
                versions.append((int(ver_str), f))
            except ValueError:
                print(f" Invalid file version format: {f.name}")
        
        if not versions:
            # Fallback: create new file
            ver = self.catalog.get_next_version(logical_key)
            return self._write_file(logical_key, ver, new_data, new_total)
        
        # Get latest file
        latest_file = max(versions)[1]
        
        # Read-merge-write
        existing_data = pq.read_table(latest_file)
        merged_data = pa.concat_tables([existing_data, new_data], promote=True)
        
        # Write new version
        ver = self.catalog.get_next_version(logical_key)
        output_path = self.base_path / f"{logical_key}_v{ver}.parquet"
        pq.write_table(merged_data, output_path, compression="zstd")
        
        # Update metadata
        self.chunk_metadata.update_row_count(logical_key, new_total)


    # --- Scan Path ---

    def scan(
        self,
        filters=[],
        columns=None,
        return_candidates=False,
        return_valid_ids=False,
        vector_ops: Optional[List[Callable]] = None,
    ) -> pa.Table:
        """
        Execute a scan with filter pushdown and vector operations.
        """
        t0 = time.time()
        tid = self.sequence.allocate_tids()
        self.active_tids[tid] = t0

        # 1. Pruning & column determination
        candidates, needed_columns = self._get_candidates(filters, columns, vector_ops)

        if not candidates:
            del self.active_tids[tid]
            empty = pa.Table.from_pylist([])
            return self._return_scan_result(
                empty, return_candidates, return_valid_ids, candidates, None
            )

        global_id = self.schema.column_map["global_id"]
        needed_columns = self._ensure_global_id(needed_columns, global_id)

        # 2. Filter phase (if needed)
        valid_ids, ids_by_chunk, cached_chunks, processed_chunks = (
            self._execute_filter_phase(
                candidates, filters, needed_columns, vector_ops, tid, global_id
            )
        )

        if valid_ids is not None and len(valid_ids) == 0:
            del self.active_tids[tid]
            empty = pa.Table.from_pylist([])
            return self._return_scan_result(
                empty, return_candidates, return_valid_ids, candidates, valid_ids
            )

        # 3. Data phase
        chunks_with_metadata = self._execute_data_phase(
            candidates,
            ids_by_chunk,
            cached_chunks,
            processed_chunks,
            needed_columns,
            vector_ops,
            tid,
            global_id,
        )

        if not chunks_with_metadata:
            del self.active_tids[tid]
            empty = pa.Table.from_pylist([])
            return self._return_scan_result(
                empty, return_candidates, return_valid_ids, candidates, valid_ids
            )

        # 4. Join & finalize
        full = self._join_chunks(chunks_with_metadata)

        # 5. Final column selection
        if columns:
            full = self._apply_final_projection(full, columns, global_id)

        del self.active_tids[tid]
        return self._return_scan_result(
            full, return_candidates, return_valid_ids, candidates, valid_ids
        )

    def _ensure_global_id(self, needed_columns: List[str], global_id: str) -> List[str]:
        """Ensure global_id is in needed columns."""
        if global_id not in needed_columns:
            return needed_columns + [global_id]
        return needed_columns

    def _execute_filter_phase(
        self, candidates, filters, needed_columns, vector_ops, tid, global_id
    ):
        """Execute filter phase and return valid IDs."""
        valid_ids = None
        ids_by_chunk = None
        cached_chunks = {}
        processed_chunks = set()

        if not filters:
            return valid_ids, ids_by_chunk, cached_chunks, processed_chunks

        filter_columns = FilterEngine.extract_columns(filters)
        common_cols = set(needed_columns) & set(filter_columns)
        can_cache = len(common_cols) > 0

        target_filter_chunks = self._get_filter_chunk_indices(filter_columns)
        has_data_filter = len(target_filter_chunks) > 0

        filter_results = self._process_filter_chunks(
            candidates,
            filters,
            filter_columns,
            needed_columns,
            target_filter_chunks,
            has_data_filter,
            can_cache,
            common_cols,
            cached_chunks,
            processed_chunks,
            vector_ops,
            tid,
            global_id,
        )

        if not filter_results:
            return None, None, cached_chunks, processed_chunks

        valid_ids = np.unique(np.concatenate(filter_results))
        ids_by_chunk = self._group_ids_by_chunk(valid_ids)

        return valid_ids, ids_by_chunk, cached_chunks, processed_chunks

    def _get_filter_chunk_indices(self, filter_columns: Set[str]) -> Set[int]:
        """Get chunk indices for filter columns."""
        target_chunks = set()
        for col in filter_columns:
            c_idx = self.schema.get_chunk_index(col, self.chunk_cols)
            if c_idx is not None:
                target_chunks.add(c_idx)
        return target_chunks

    def _group_ids_by_chunk(self, valid_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """Group valid IDs by row chunk."""
        ids_by_chunk = {}
        chunk_indices = valid_ids // self.chunk_rows
        unique_chunks = np.unique(chunk_indices)

        for chk_idx in unique_chunks:
            ids_by_chunk[chk_idx] = valid_ids[chunk_indices == chk_idx]

        return ids_by_chunk

    def _process_filter_chunks(
        self,
        candidates,
        filters,
        filter_columns,
        needed_columns,
        target_filter_chunks,
        has_data_filter,
        can_cache,
        common_cols,
        cached_chunks,
        processed_chunks,
        vector_ops,
        tid,
        global_id,
    ):
        """Process chunks for filtering."""
        filter_results = []

        def process_filter(path):
            try:
                part = path.name.split("_v")[0]

                # Check cache
                cache_tid = 0
                if part in self._in_memory_updates:
                    cache_tid = min(
                        t for t in self._in_memory_updates[part] if t <= tid
                    )
                    t = self._in_memory_updates[part][cache_tid]
                else:
                    # Read from disk
                    parts = path.name.split("_")
                    chunk_c_idx = int(parts[2][1:])

                    if has_data_filter and chunk_c_idx not in target_filter_chunks:
                        return None

                    if not has_data_filter and chunk_c_idx != 0:
                        return None

                    schema = pq.read_schema(path)
                    filter_cols_needed = list(filter_columns) + [global_id]
                    avail = [c for c in filter_cols_needed if c in schema.names]

                    if not all(fc in avail for fc in filter_columns):
                        return None

                    cols_to_read = (
                        list(set(avail) | (set(needed_columns) & set(schema.names)))
                        if can_cache
                        else avail
                    )

                    t = pq.read_table(path, columns=cols_to_read)

                # Apply patches
                patches = [
                    (tid_p, p)
                    for tid_p, p in self.patch_log.items()
                    if cache_tid <= tid_p <= tid
                ]
                for patch_tid, patch_dict in patches:
                    patch = patch_dict.get(part)
                    if patch is not None:
                        t = self._update(t, patch)

                # Apply filters
                mask = FilterEngine.evaluate(t, filters)
                if mask is None or pc.sum(mask.cast("int8")).as_py() == 0:
                    return None

                filtered_t = t.filter(mask)

                # Apply vector ops if present
                if vector_ops and len(filtered_t) > 0:
                    try:
                        for op_func in vector_ops:
                            filtered_t = op_func(filtered_t)
                        processed_chunks.add(path)
                    except Exception as e:
                        print(f"⚠️ Error in vector op: {e}")

                # Cache if useful
                if can_cache:
                    chunk_useful_cols = (
                        set(filtered_t.column_names) - {global_id}
                    ) & common_cols
                    if chunk_useful_cols:
                        cached_chunks[path] = filtered_t

                return filtered_t[global_id].to_numpy()

            except Exception as e:
                print(f"⚠️ Error processing filter: {e}")
                return None

        futures = [self.io_pool.submit(process_filter, p) for p in candidates]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                filter_results.append(res)

        return filter_results

    def _execute_data_phase(
        self,
        candidates,
        ids_by_chunk,
        cached_chunks,
        processed_chunks,
        needed_columns,
        vector_ops,
        tid,
        global_id,
    ):
        """Execute data retrieval phase."""
        chunks_with_metadata = []

        def process_data(path):
            try:
                logical_key = path.name.split("_v")[0]

                # Get row chunk index
                parts = path.name.split("_")
                row_chunk_idx = int(parts[1][1:])

                # Get valid IDs for this chunk
                chunk_valid_ids = (
                    ids_by_chunk.get(row_chunk_idx) if ids_by_chunk else None
                )

                if ids_by_chunk is not None and chunk_valid_ids is None:
                    return None

                # Check for pending patches
                has_pending_patches = any(
                    logical_key in self.patch_log.get(ptid, {})
                    for ptid in self.patch_log.keys()
                    if ptid <= tid
                )

                # Try cache first
                table = self._try_load_from_cache(
                    logical_key, tid, chunk_valid_ids, needed_columns, global_id
                )

                if table is None:
                    # Load from disk
                    table = self._load_from_disk(
                        path,
                        logical_key,
                        cached_chunks,
                        has_pending_patches,
                        needed_columns,
                        chunk_valid_ids,
                        global_id,
                        tid,
                    )

                if table is None:
                    return None

                # Apply vector ops if needed
                if vector_ops and len(table) > 0 and path not in processed_chunks:
                    try:
                        for op_func in vector_ops:
                            table = op_func(table)
                        processed_chunks.add(path)
                    except Exception as e:
                        print(f"⚠️ Error in vector ops: {e}")

                return (path, table)

            except Exception as e:
                print(f"⚠️ Error in process_data: {e}")
                return None

        futures = [self.io_pool.submit(process_data, p) for p in candidates]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                chunks_with_metadata.append(res)

        return chunks_with_metadata

    def _try_load_from_cache(
        self, logical_key, tid, chunk_valid_ids, needed_columns, global_id
    ):
        """Try loading from in-memory cache."""
        with self._cache_lock:
            if logical_key not in self._in_memory_updates:
                return None

            available_tids = [
                t for t in self._in_memory_updates[logical_key] if t <= tid
            ]
            if not available_tids:
                return None

            cached_tid = max(available_tids)
            cached_table = self._in_memory_updates[logical_key][cached_tid]

            # Apply subsequent patches
            for patch_tid in range(cached_tid + 1, tid + 1):
                if (
                    patch_tid in self.patch_log
                    and logical_key in self.patch_log[patch_tid]
                ):
                    cached_table = self._update(
                        cached_table, self.patch_log[patch_tid][logical_key]
                    )

            # Filter by valid IDs
            if chunk_valid_ids is not None:
                cached_ids = cached_table[global_id].to_numpy()
                mask = np.isin(cached_ids, chunk_valid_ids)
                if not mask.any():
                    return None
                cached_table = cached_table.filter(pa.array(mask))

            # Select columns
            available_cols = set(cached_table.column_names)
            keep_cols = [c for c in needed_columns if c in available_cols]

            return cached_table.select(keep_cols)

    def _load_from_disk(
        self,
        path,
        logical_key,
        cached_chunks,
        has_pending_patches,
        needed_columns,
        chunk_valid_ids,
        global_id,
        tid,
    ):
        """Load chunk from disk."""
        if path in cached_chunks:
            table = cached_chunks[path]
        else:
            schema = pq.read_schema(path)

            if has_pending_patches:
                # Full read for consistency
                table = pq.read_table(path)
            else:
                # Optimized read
                avail = [c for c in needed_columns if c in schema.names]
                if not avail:
                    return None
                table = pq.read_table(path, columns=avail)

        # Filter by valid IDs
        if chunk_valid_ids is not None:
            table_ids = table[global_id].to_numpy()
            mask = np.isin(table_ids, chunk_valid_ids)
            if not mask.any():
                return None
            table = table.filter(pa.array(mask))

        # Apply patches
        if has_pending_patches:
            for patch_tid in sorted(self.patch_log.keys()):
                if patch_tid > tid:
                    break

                if logical_key in self.patch_log[patch_tid]:
                    table = self._update(table, self.patch_log[patch_tid][logical_key])

            # Cache complete chunk
            with self._cache_lock:
                if logical_key not in self._in_memory_updates:
                    self._in_memory_updates[logical_key] = {}
                self._in_memory_updates[logical_key][tid] = table

        return table

    def _apply_final_projection(
        self, table: pa.Table, columns: List[str], global_id: str
    ) -> pa.Table:
        """Apply final column projection."""
        select_cols = columns if global_id in columns else columns + [global_id]
        available_cols = [c for c in select_cols if c in table.column_names]

        # Include computed columns
        original_schema_cols = set(self.schema.column_map.keys())
        computed_cols = [
            c
            for c in table.column_names
            if c not in original_schema_cols and c != global_id
        ]

        final_cols = list(dict.fromkeys(available_cols + computed_cols))
        return table.select(final_cols)

    def _return_scan_result(
        self, table, return_candidates, return_valid_ids, candidates, valid_ids
    ):
        """Return scan result with optional metadata."""
        if return_candidates and return_valid_ids:
            return (
                table,
                candidates,
                valid_ids if valid_ids is not None else np.array([]),
            )
        if return_candidates:
            return table, candidates
        if return_valid_ids:
            return table, valid_ids if valid_ids is not None else np.array([])
        return table

    # --- Join Logic ---

    def _join_chunks(self, chunks_with_metadata):
        """Join chunks with vertical concat and horizontal join."""
        if not chunks_with_metadata:
            return pa.Table.from_pylist([])

        if len(chunks_with_metadata) == 1:
            return chunks_with_metadata[0][1]

        global_id = self.schema.column_map["global_id"]

        # Group by vertical chunk index
        chunks_by_vertical = defaultdict(list)
        for path, chunk in chunks_with_metadata:
            try:
                parts = path.name.split("_")
                c_idx = int(parts[2][1:])
                chunks_by_vertical[c_idx].append((path, chunk))
            except Exception:
                chunks_by_vertical[-1].append((path, chunk))

        # Vertical concat within same c_idx
        merged_tables = []
        for c_idx, chunk_list in chunks_by_vertical.items():
            if len(chunk_list) == 1:
                merged_tables.append(chunk_list[0][1])
            else:
                tables_to_concat = [chunk for _, chunk in chunk_list]
                merged_tables.append(pa.concat_tables(tables_to_concat, promote=True))

        if len(merged_tables) == 1:
            return merged_tables[0]

        # Horizontal join
        result = merged_tables[0]
        for next_table in merged_tables[1:]:
            result = self._horizontal_join(result, next_table, global_id)

        return result

    def _horizontal_join(
        self, left: pa.Table, right: pa.Table, join_key: str
    ) -> pa.Table:
        """Inner join between two PyArrow tables."""
        cols_to_join = [c for c in right.column_names if c != join_key]

        if not cols_to_join:
            return left

        left_ids = left[join_key].to_numpy()
        right_ids = right[join_key].to_numpy()

        left_id_to_idx = {gid: i for i, gid in enumerate(left_ids)}
        right_id_to_idx = {gid: i for i, gid in enumerate(right_ids)}

        common_ids = set(left_ids) & set(right_ids)

        if not common_ids:
            return pa.Table.from_pylist([])

        common_ids_sorted = sorted(common_ids)

        left_indices = [left_id_to_idx[gid] for gid in common_ids_sorted]
        right_indices = [right_id_to_idx[gid] for gid in common_ids_sorted]

        aligned_left = left.take(left_indices)
        aligned_right = right.take(right_indices)

        for col in cols_to_join:
            aligned_left = aligned_left.append_column(col, aligned_right[col])

        return aligned_left

    # --- Pruning ---

    def _get_candidates(self, filters, columns=[], vector_ops=None):
        """Get candidate files with pruning."""
        candidates = self.catalog.get_active_files()

        # Extract columns from vector ops
        vector_columns = set()
        if vector_ops:
            for op in vector_ops:
                if hasattr(op, "columns"):
                    vector_columns.update(op.columns)

        if filters:
            candidates = self._evaluate_pruning(filters, set(candidates))
            filter_columns = FilterEngine.extract_columns(filters)
            columns = list(set(columns) | filter_columns | vector_columns)
        else:
            columns = list(set(columns) | vector_columns)

        if columns:
            target_chunks = set()
            for col in columns:
                idx = self.schema.get_chunk_index(col, self.chunk_cols)
                if idx is None:
                    continue
                target_chunks.add(idx)

            final = []
            for p in candidates:
                try:
                    if int(p.name.split("_")[2][1:]) in target_chunks:
                        final.append(p)
                except:
                    final.append(p)
            return final, columns

        return list(candidates), columns

    def _evaluate_pruning(self, expression, candidates):
        """Evaluate pruning expression."""
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
        """Filter candidates by leaf condition."""
        col, op, val = cond

        if op != "=":
            return candidates

        tr, th, trg = None, {}, {}

        if col == self.schema.column_map["global_id"]:
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

            if "h" in p.name:
                h_vals = [int(x) for x in parts[3][1:].split("-")]
                if any(h_vals[i] != b for i, b in th.items()):
                    continue

            if "rg" in p.name:
                r_vals = [int(x) for x in parts[4][1:].split("-")]
                if any(r_vals[i] != b for i, b in trg.items()):
                    continue

            filtered.add(p)

        return filtered

    # --- Update Path ---

    def update(self, filters, updates, unsafe=False, vector_ops=None):
        """
        Write patch log for merge-on-read updates.
        """
        tid = self.sequence.allocate_tids()

        candidates, _ = self._get_candidates(filters, columns=list(updates.keys()))
        final_candidates = self._evaluate_pruning(filters, set(candidates))

        if not final_candidates:
            print(f" Update TID={tid}: No chunks found")
            return

        serialized_vector_ops = VectorOpSerializer.serialize_ops(vector_ops)

        if tid not in self.patch_log:
            self.patch_log[tid] = {}

        for candidate in final_candidates:
            logical_key = candidate.name.split("_v")[0]

            thread_id = threading.get_ident()
            process_id = os.getpid()
            patch_file = (
                self.base_path
                / f"{logical_key}_patch_TID{tid}_TH{thread_id}_PID{process_id}.json"
            )

            patch_data = {
                "tid": tid,
                "thread_id": thread_id,
                "process_id": process_id,
                "filters": filters,
                "updates": updates,
                "unsafe": unsafe,
                "vector_ops": serialized_vector_ops,
            }

            self.patch_log[tid][logical_key] = patch_data

            with open(patch_file, "w") as f:
                json.dump(patch_data, f, indent=2, cls=NumpyEncoder)

        print(
            f"✅ Update TID={tid}: Patch log saved for {len(final_candidates)} chunk(s)"
        )

    def _update(
        self,
        table: pa.Table,
        patch_metadata: Dict,
        logical_key: str = None,
        candidate_path: Path = None,
    ) -> pa.Table:
        """
        Apply a single patch to an in-memory table.
        """
        filters = patch_metadata["filters"]
        updates = patch_metadata["updates"]

        # Deserialize vector ops
        serialized_vector_ops = patch_metadata.get("vector_ops")
        vector_ops = VectorOpSerializer.deserialize_ops(serialized_vector_ops)

        # Check for missing columns
        missing_update_cols = set(updates.keys()) - set(table.column_names)
        missing_vector_cols = set()

        if vector_ops:
            for op in vector_ops:
                if hasattr(op, "columns"):
                    missing_vector_cols.update(
                        set(op.columns) - set(table.column_names)
                    )

        missing_cols = missing_update_cols | missing_vector_cols

        # Reload if missing critical columns
        if missing_cols and candidate_path and candidate_path.exists():
            try:
                full_table = pq.read_table(candidate_path)

                # Apply previous patches
                tid = patch_metadata.get("tid", 0)
                for patch_tid in sorted(self.patch_log.keys()):
                    if patch_tid >= tid:
                        break
                    if logical_key in self.patch_log[patch_tid]:
                        full_table = self._update(
                            full_table,
                            self.patch_log[patch_tid][logical_key],
                            logical_key=logical_key,
                            candidate_path=candidate_path,
                        )

                table = full_table
            except Exception as e:
                print(f" Error reloading chunk: {e}")

        # Apply vector ops
        if vector_ops:
            try:
                for op_func in vector_ops:
                    table = op_func(table)
            except Exception as e:
                print(f" Error in vector ops: {e}")

        # Calculate mask
        mask = FilterEngine.evaluate(table, filters)
        if mask is None or pc.sum(mask.cast("int8")).as_py() == 0:
            return table

        # Apply updates
        new_cols = []
        for col in table.column_names:
            if col in updates:
                update_val = updates[col]

                if callable(update_val):
                    new_val = update_val(table)
                else:
                    new_val = pa.scalar(update_val, type=table[col].type)

                new_cols.append(pc.if_else(mask, new_val, table[col]))
            else:
                new_cols.append(table[col])

        return pa.Table.from_arrays(new_cols, names=table.column_names)

    
    # DEPRECATED TODO IMPLEMENT NEW COMPACTION
    def vacuum(self):
        """Clean up old versions (deprecated)."""
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
        """Close storage and cleanup resources."""
        self.io_pool.shutdown()
