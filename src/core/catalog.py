from pathlib import Path
import threading
from typing import Dict, List


class FileCatalog:
    # Really RAW  version
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.active_versions: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.refresh()

    def refresh(self):
        with self.lock:
            self.active_versions.clear()
            if not self.base_path.exists():
                return
            for f in self.base_path.iterdir():
                if not f.name.endswith(".parquet"):
                    continue
                try:
                    idx_v = f.name.rfind("_v")
                    if idx_v == -1:
                        continue
                    logical_key = f.name[:idx_v]
                    ver = int(f.name[idx_v + 2 : -8])
                    curr = self.active_versions.get(logical_key, 0)
                    if ver > curr:
                        self.active_versions[logical_key] = ver
                except:
                    continue

    def get_next_version(self, logical_key: str) -> int:
        with self.lock:
            new_v = self.active_versions.get(logical_key, 0) + 1
            self.active_versions[logical_key] = new_v
            return new_v

    def get_active_files(self) -> List[Path]:
        with self.lock:
            return [
                self.base_path / f"{k}_v{v}.parquet"
                for k, v in self.active_versions.items()
            ]
