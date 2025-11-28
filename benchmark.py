
import time
import numpy as np
import pyarrow as pa
import shutil
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.storage import NDimStorage


TOTAL_ROWS = 1_000_000
TOTAL_COLS = 128
CHUNK_ROWS = 100_000
CHUNK_COLS = 32  

PATH = "./htap_production_data"


def generate_wide_batch(rows):
    """Generates a wide dataset."""
    data = {
        "region": np.random.randint(0, 10, rows).astype(np.int32),
        "price": np.random.randint(0, 1000, rows).astype(np.int32),
        "status": np.random.choice(["active", "inactive", "banned"], rows),
        "category": np.random.choice(["A", "B", "C", "D"], rows),
    }
    # Add metric columns
    matrix = np.random.rand(rows, TOTAL_COLS).astype(np.float32)
    for i in range(TOTAL_COLS):
        data[f"metric_{i}"] = matrix[:, i]
    return pa.Table.from_pydict(data)


def run_suite():

    if os.path.exists(PATH):
        shutil.rmtree(PATH)
        time.sleep(0.5)

    print(f"--- INITIALIZATION (1M Rows, {TOTAL_COLS} Cols) ---")
    store = NDimStorage(
        PATH,
        chunk_rows=CHUNK_ROWS,
        chunk_cols=CHUNK_COLS,
        hash_dims={"region": 10},
        range_dims={"price": 250},
    )
    print("\n--- 1. INGESTION ---")
    t_start = time.time()
    batch_size = 250_000
    for i in range(0, TOTAL_ROWS, batch_size):
        print(f"   Writing batch {i//batch_size + 1}...")
        table = generate_wide_batch(batch_size)
        store.write_batch(table)
    print(f" Total Ingestion Time: {time.time() - t_start:.2f}s")

    print("\n--- 2. VERTICAL PRUNING TEST ---")
    print("Query: SELECT metric_0, metric_127 WHERE region=1")

    t0 = time.time()
    res = store.scan([("region", "=", 1)], columns=["metric_0", "metric_127"])
    dt = time.time() - t0

    print(f" Result: {len(res)} rows")
    print(f" Time: {dt:.4f}s")

    print("\n--- 3. OLTP POINT LOOKUP ---")
    target_id = TOTAL_ROWS - 500
    print(f"Looking for Global ID: {target_id}")

    t0 = time.time()
    res = store.scan([("global_id", "=", target_id)], columns=["status"])
    print(f" Time: {time.time() - t0:.4f}s")
    print(f"Found: {len(res)}")

    print("\n--- 4. COMPLEX LOGIC UPDATE ---")
    print("Query: UPDATE metric_0 = -99.9 WHERE (region=2 OR region=3) AND price > 800")

    filters = [
        [("region", "=", 2), "OR", ("region", "=", 3)],
        "AND",
        ("price", ">", 800),
    ]

    store.update(filters, {"metric_0": -99.9})

    # Verify
    chk = store.scan(filters, columns=["metric_0"])
    if len(chk) > 0:
        sample = chk["metric_0"][0].as_py()
        print(f"   Verification Sample: {sample} (Expected -99.9)")

    # 6. VACUUM
    store.vacuum()
    store.close()


if __name__ == "__main__":
    run_suite()
