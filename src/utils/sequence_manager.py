from pathlib import Path
import threading
import json


class SequenceManager:
    # TODO: Implement persistent sequence storage, NOT PRODUCTION READY
    """Manages a persistent auto-increment counter for Global ID."""

    def __init__(self, base_path: Path):
        self.path = base_path / "sequence.json"
        self.lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            with open(self.path, "w") as f:
                json.dump({"next_id": 0}, f)

    def allocate_ids(self, count: int) -> int:
        """Allocates a batch of IDs atomically."""
        with self.lock:
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
            except:
                data = {"next_id": 0}

            start_id = data.get("next_id", 0)
            data["next_id"] = start_id + count

            with open(self.path, "w") as f:
                json.dump(data, f)
            return start_id
