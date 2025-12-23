from pathlib import Path
from typing import Dict, List
import json
import fcntl
import contextlib
import os


class Lockmanager:
    """File-based lock manager for cross-process synchronization."""

    def __init__(self, lock_file: Path, open_mode: str = "a+"):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        self.file_handle = open(self.lock_file, open_mode)

    @contextlib.contextmanager
    def acquire(self):
        try:
            fcntl.flock(self.file_handle, fcntl.LOCK_EX)
            yield
        finally:

            fcntl.flock(self.file_handle, fcntl.LOCK_UN)


class SequenceManager:
    """Manages a persistent auto-increment counter for Global ID."""

    def __init__(self, base_path: Path):
        self.id_path = base_path / "global_id_sequence.json"
        self.tid_path = base_path / "global_tid_sequence.json"

        self.global_id_lock = Lockmanager(self.id_path)
        self.global_tid_lock = Lockmanager(self.tid_path)

    def _atomic_allocate(
        self, lock_manager: Lockmanager, path: Path, key: str, count: int = 1
    ) -> int:
        """Metodo privato per allocare atomicamente ID da un file specifico."""

        with lock_manager.acquire():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {key: 0}

            start_id = data.get(key, 0)
            data[key] = start_id + count

            with open(path, "w") as f:
                json.dump(data, f, indent=4)

                f.flush()
                os.fsync(f.fileno())

            return start_id

    def allocate_ids(self, count: int = 1) -> int:
        """Allocates un batch di Global ID per il chunking."""
        return self._atomic_allocate(
            lock_manager=self.global_id_lock,
            path=self.id_path,
            key="next_id",
            count=count,
        )

    def allocate_tids(self) -> int:
        """Allocates un singolo Global Transaction ID (TID) per la patch log."""
        return self._atomic_allocate(
            lock_manager=self.global_tid_lock,
            path=self.tid_path,
            key="next_global_tid",
            count=1,
        )

    def current_tid(self) -> int:
        """Restituisce l'ultimo TID allocato (senza incrementarlo)."""
        try:
            with open(self.tid_path, "r") as f:
                data = json.load(f)
                return data.get("next_global_tid", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0


class FileCatalog:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.catalog_path = base_path / "catalog.json"

        self.lock_path = base_path / "catalog.lock"
        self.lock_manager = Lockmanager(self.lock_path)

        self._active_versions: Dict[str, int] = {}
        self._load()

    # --- Persistence Methods ---

    def _load(self):
        """Loads the catalog state from disk."""
        try:
            with open(self.catalog_path, "r") as f:
                self._active_versions = json.load(f).get("versions", {})
        except (FileNotFoundError, json.JSONDecodeError):
            self._active_versions = {}

    def _save_atomic(self, versions: Dict[str, int]):
        """Atomically saves the catalog state to disk with fsync."""
        data = {"versions": versions}

        with open(self.catalog_path, "w") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

    # --- Public Methods ---

    def refresh(self):
        """Refreshes the in-memory cache by loading the latest state from the disk file."""
        self._load()

    def get_next_version(self, logical_key: str) -> int:
        """Atomically allocates the next version for a logical key and persists the change."""
        with self.lock_manager.acquire():
            self._load()
            new_v = self._active_versions.get(logical_key, 0) + 1
            self._active_versions[logical_key] = new_v
            self._save_atomic(self._active_versions)
            return new_v

    def get_active_files(self) -> List[Path]:
        """Returns the paths of the active files based on the latest committed version."""
        self._load()
        return [
            self.base_path / f"{k}_v{v}.parquet"
            for k, v in self._active_versions.items()
        ]
