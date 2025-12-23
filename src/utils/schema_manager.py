import json
import threading
import queue

from pathlib import Path
from typing import Dict



class ChunkMetadataManager:
    """
    In-memory cached metadata with async disk persistence.
    - Reads: Lock-free from memory
    - Writes: In-memory + queued for background flush
    - Consistency: Periodic + shutdown flush
    """
    
    def __init__(self, base_path: Path, flush_interval: float = 2.0):
        self.path = base_path / "chunk_metadata.json"
        self.flush_interval = flush_interval
        
        # In-memory cache (read-optimized)
        self._metadata: Dict[str, int] = {}
        self._lock = threading.RLock()  # Read-write lock simulation
        
        # Dirty tracking for incremental saves
        self._dirty_keys: set = set()
        self._dirty_lock = threading.Lock()
        
        # Background persistence
        self._flush_queue = queue.Queue()
        self._shutdown = threading.Event()
        self._flush_thread = threading.Thread(target=self._background_flusher, daemon=True)
        self._flush_thread.start()
        
        # Load initial state
        self._load()
    
    def _load(self):
        """Load from disk (startup only)."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self._metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._metadata = {}
    
    def _save(self):
        """Synchronous save (blocking, only for shutdown)."""
        with open(self.path, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    def _background_flusher(self):
        """Background thread that periodically flushes dirty data."""
        while not self._shutdown.is_set():
            try:
                # Wait for flush signal or timeout
                self._flush_queue.get(timeout=self.flush_interval)
            except queue.Empty:
                pass
            
            # Check if there's dirty data
            with self._dirty_lock:
                if not self._dirty_keys:
                    continue
                
                # Snapshot dirty keys and clear
                dirty_snapshot = self._dirty_keys.copy()
                self._dirty_keys.clear()
            
            # Incremental save (only dirty data)
            # For simplicity, we save full dict but could optimize
            with self._lock:
                snapshot = self._metadata.copy()
            
            try:
                with open(self.path, "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception as e:
                print(f"⚠️ Metadata flush failed: {e}")
                # Re-mark as dirty
                with self._dirty_lock:
                    self._dirty_keys.update(dirty_snapshot)
    
    # --- Public API (unchanged from outside) ---
    
    def get_row_count(self, logical_key: str) -> int:
        """Fast read from in-memory cache (no I/O)."""
        with self._lock:
            return self._metadata.get(logical_key, 0)
    
    def update_row_count(self, logical_key: str, new_count: int):
        """Fast write to memory + async persistence."""
        with self._lock:
            self._metadata[logical_key] = new_count
        
        # Mark as dirty for background flush
        with self._dirty_lock:
            self._dirty_keys.add(logical_key)
        
        # Signal flush thread (non-blocking)
        try:
            self._flush_queue.put_nowait(None)
        except queue.Full:
            pass  # Already queued
    
    def flush(self):
        """Force immediate flush (blocking)."""
        with self._lock:
            snapshot = self._metadata.copy()
        
        with open(self.path, "w") as f:
            json.dump(snapshot, f, indent=2)
        
        with self._dirty_lock:
            self._dirty_keys.clear()
    
    def shutdown(self):
        """Graceful shutdown with final flush."""
        self._shutdown.set()
        self.flush()
        self._flush_thread.join(timeout=5.0)


class SchemaManager:
    """
    Optimized with in-memory cache + async persistence.
    Same interface, better performance.
    """
    
    def __init__(self, base_path: Path, flush_interval: float = 2.0):
        self.path = base_path / "schema.json"
        self.flush_interval = flush_interval
        
        # In-memory state
        self.column_map: Dict[str, int] = {}
        self.metadata: Dict[str, Dict[str, any]] = {}
        self._lock = threading.RLock()
        
        # Dirty tracking
        self._dirty = False
        self._dirty_lock = threading.Lock()
        
        # Background flush
        self._flush_queue = queue.Queue()
        self._shutdown = threading.Event()
        self._flush_thread = threading.Thread(target=self._background_flusher, daemon=True)
        self._flush_thread.start()
        
        self._load()
    
    def _load(self):
        """Load from disk (startup)."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    self.column_map = data.get("column_map", {})
                    self.metadata = data.get("metadata", {})
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    
    def _save(self):
        """Synchronous save (blocking)."""
        with self._lock:
            data = {
                "column_map": self.column_map,
                "metadata": self.metadata
            }
        
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _background_flusher(self):
        """Background persistence thread."""
        while not self._shutdown.is_set():
            try:
                self._flush_queue.get(timeout=self.flush_interval)
            except queue.Empty:
                pass
            
            with self._dirty_lock:
                if not self._dirty:
                    continue
                self._dirty = False
            
            try:
                self._save()
            except Exception as e:
                print(f"⚠️ Schema flush failed: {e}")
                with self._dirty_lock:
                    self._dirty = True
    
    def update(self, column_names, global_id=None, table=None):
        """Fast update with deferred persistence."""
        with self._lock:
            changed = False
            
            self.column_map["global_id"] = global_id if global_id else "global_id"
            
            current_max_idx = max(
                [v for k, v in self.column_map.items() if k != "global_id"],
                default=-1
            )
            
            data_cols = [c for c in column_names if c != self.column_map["global_id"]]
            
            for col in data_cols:
                if col not in self.column_map and col != "global_id":
                    current_max_idx += 1
                    self.column_map[col] = current_max_idx
                    changed = True
                    
                    if table is not None and col in table.column_names:
                        self._infer_metadata(col, table[col].type, table[col])
            
            if changed:
                with self._dirty_lock:
                    self._dirty = True
                
                try:
                    self._flush_queue.put_nowait(None)
                except queue.Full:
                    pass
    
    def _infer_metadata(self, col_name, arrow_type, column):
        """Same as before (runs under self._lock)."""
        import pyarrow as pa
        
        metadata = {}
        
        if pa.types.is_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
            metadata["type"] = "array"
            
            value_type = arrow_type.value_type
            if pa.types.is_floating(value_type):
                metadata["dtype"] = "float32" if value_type == pa.float32() else "float64"
            elif pa.types.is_integer(value_type):
                metadata["dtype"] = "int32" if value_type == pa.int32() else "int64"
            else:
                metadata["dtype"] = str(value_type)
            
            if pa.types.is_fixed_size_list(arrow_type):
                metadata["shape"] = [arrow_type.list_size]
            else:
                combined = column.combine_chunks()
                if len(combined) > 0:
                    first_elem = combined[0].as_py()
                    if first_elem is not None:
                        metadata["shape"] = [len(first_elem)]
        else:
            metadata["type"] = "scalar"
            metadata["dtype"] = str(arrow_type)
        
        self.metadata[col_name] = metadata
    
    def get_metadata(self, col_name):
        with self._lock:
            return self.metadata.get(col_name)
    
    def get_array_shape(self, col_name):
        with self._lock:
            meta = self.metadata.get(col_name)
            if meta and meta.get("type") == "array":
                return meta.get("shape")
        return None
    
    def get_chunk_index(self, col_name, chunk_size):
        with self._lock:
            if col_name == self.column_map["global_id"]:
                return 0
            
            if col_name not in self.column_map:
                return None
            
            col_idx = self.column_map[col_name]
            return col_idx // chunk_size
    
    def flush(self):
        """Force immediate flush."""
        self._save()
        with self._dirty_lock:
            self._dirty = False
    
    def shutdown(self):
        """Graceful shutdown."""
        self._shutdown.set()
        self.flush()
        self._flush_thread.join(timeout=5.0)
    
    # Backward compatibility property
    @property
    def lock(self):
        return self._lock