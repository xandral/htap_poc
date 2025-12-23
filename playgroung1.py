
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import shutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.storage import NDimStorage

# ==========================================
# CONFIGURATION
# ==========================================
TOTAL_ROWS = 1_000_000
TOTAL_COLS = 128
CHUNK_ROWS = 100_000
CHUNK_COLS = 32
PATH = "./htap_ultimate_data"


def generate_mixed_batch(rows):
    """
    Generates comprehensive test data supporting all benchmark scenarios.
    """
    data = {
        "region": np.random.randint(0, 10, rows).astype(np.int32),
        "price": np.random.randint(0, 1000, rows).astype(np.int32),
        "status": np.random.choice(["active", "inactive", "banned"], rows),
        "category": np.random.choice(["A", "B", "C", "D"], rows),
    }

    matrix = np.random.rand(rows, TOTAL_COLS).astype(np.float32)

    # Wide-table style: 128 separate columns
    for i in range(TOTAL_COLS):
        data[f"metric_{i}"] = matrix[:, i]

    # SQL MDA style: array column
    data["sensor_matrix"] = list(matrix)

    return pa.Table.from_pydict(data)


# ==========================================
# BENCHMARK SUITE
# ==========================================
def run_ultimate_benchmark():
    """Main benchmark orchestrator."""

    # Setup
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
        time.sleep(0.5)

    print("=" * 80)
    print("HTAP ULTIMATE BENCHMARK SUITE - Merge-on-Read Edition")
    print("=" * 80)
    print(
        f"Configuration: {TOTAL_ROWS:,} rows | {TOTAL_COLS} cols | "
        f"Chunk: {CHUNK_ROWS:,}x{CHUNK_COLS}"
    )
    print("=" * 80)

    store = NDimStorage(
        PATH,
        chunk_rows=CHUNK_ROWS,
        chunk_cols=CHUNK_COLS,
        hash_dims={"region": 10},
        range_dims={"price": 250},
    )

    # ==========================================
    # PART 1: CORE HTAP OPERATIONS
    # ==========================================
    print("\n" + "=" * 80)
    print("PART 1: CORE HTAP OPERATIONS")
    print("=" * 80)

    # Test 1: Ingestion
    print("\n--- 1. INGESTION ---")
    t_start = time.time()
    batch_size = 50_000
    for i in range(0, TOTAL_ROWS, batch_size):
        batch_num = i // batch_size + 1
        print(f"   Writing batch {batch_num}/{TOTAL_ROWS//batch_size}...", end="\r")
        table = generate_mixed_batch(batch_size)
        store.write_batch(table)
    t_ingest = time.time() - t_start
    print(
        f"\n Ingestion Complete: {TOTAL_ROWS:,} rows in {t_ingest:.2f}s "
        f"({TOTAL_ROWS/t_ingest:,.0f} rows/s)"
    )

    # Test 2: OLTP Point Lookup
    print("\n--- 2. OLTP POINT LOOKUP ---")
    target_id = TOTAL_ROWS - 500
    t0 = time.time()
    res = store.scan([("global_id", "=", target_id)], columns=["status", "price"])
    t_oltp = time.time() - t0
    print(
        f" Found ID {target_id} | Time: {t_oltp*1000:.2f}ms | "
        f"Result: {len(res)} rows"
    )

    # Test 3: OLAP Aggregation (Vertical Pruning)
    print("\n--- 3. OLAP PROJECTION (SELECT metric_0, metric_127) ---")
    t0 = time.time()
    res = store.scan([("region", "=", 1)], columns=["metric_0", "metric_127", "region"])
    t_olap = time.time() - t0
    print(
        f" Projection: {len(res)} rows | Time: {t_olap:.4f}s | "
        f"Throughput: {len(res)/t_olap:,.0f} rows/s"
    )

    # Test 4: Complex Filter (Horizontal Pruning)
    print("\n--- 4. COMPLEX FILTER (Multi-Condition) ---")
    print("Query: WHERE (region=2 OR region=3) AND price > 800")
    t0 = time.time()
    filters = [
        [("region", "=", 2), "OR", ("region", "=", 3)],
        "AND",
        ("price", ">", 800),
    ]
    res = store.scan(filters, columns=["region", "price", "status"])
    t_filter = time.time() - t0
    print(f" Filtered: {len(res)} rows | Time: {t_filter:.4f}s")

    # ==========================================
    # PART 2: MERGE-ON-READ CORE TESTS
    # ==========================================
    print("\n" + "=" * 80)
    print("PART 2: MERGE-ON-READ TESTS")
    print("=" * 80)

    # Test 5: Update Latency
    print("\n--- 5. UPDATE LATENCY (Patch Log Write) ---")
    TARGET_REGION = 5
    NEW_VALUE = np.float32(999.0)

    # Pre-scan baseline
    t0 = time.time()
    pre_scan = store.scan(
        [("region", "=", TARGET_REGION)], columns=["region", "metric_0"]
    )
    target_count = len(pre_scan)
    t_prescan = time.time() - t0
    print(f"   Pre-scan: {target_count:,} rows | Time: {t_prescan:.4f}s")

    # Fast update (only writes patch log)
    t0 = time.time()
    store.update(
        filters=[("region", "=", TARGET_REGION)], updates={"metric_0": NEW_VALUE}
    )
    t_update = time.time() - t0
    print(
        f" Update: {target_count:,} rows | Time: {t_update*1000:.2f}ms | "
        f"Speedup: {t_prescan/t_update:.1f}x faster than scan"
    )

    # Test 6: Immediate Consistency (Read-after-Write)
    print("\n--- 6. IMMEDIATE CONSISTENCY (Read-After-Write) ---")
    t0 = time.time()
    result = store.scan(
        filters=[("metric_0", "=", NEW_VALUE)], columns=["region", "metric_0", "price"]
    )
    t_mor_scan = time.time() - t0
    found_count = len(result)

    print(f" MoR Scan: {found_count:,} rows | Time: {t_mor_scan:.4f}s")
    if found_count == target_count:
        print("    SUCCESS: All updated records visible (Immediate Consistency OK)")
    else:
        print(f"    FAILURE: Expected {target_count}, found {found_count}")

    # Test 7: Cache Performance
    print("\n--- 7. CACHE PERFORMANCE (Second Read) ---")
    t0 = time.time()
    result2 = store.scan(
        filters=[("metric_0", "=", NEW_VALUE)], columns=["region", "metric_0"]
    )
    t_cached = time.time() - t0
    speedup = t_mor_scan / t_cached
    print(
        f" Cached Scan: {len(result2):,} rows | Time: {t_cached:.4f}s | "
        f"Speedup: {speedup:.2f}x"
    )

    # Test 8: Multi-Patch Overhead
    print("\n--- 8. MULTI-PATCH OVERHEAD (5 Consecutive Updates) ---")
    TARGET_REGION_2 = 7
    NUM_UPDATES = 5

    # Baseline
    pre_multi = store.scan(
        [("region", "=", TARGET_REGION_2)], columns=["price", "status"]
    )
    multi_count = len(pre_multi)
    print(f"   Target: {multi_count:,} rows with region={TARGET_REGION_2}")

    # Sequential updates
    update_times = []
    for i in range(1, NUM_UPDATES + 1):
        t0 = time.time()
        store.update(
            filters=[("region", "=", TARGET_REGION_2)],
            updates={"status": f"updated_{i}"},
        )
        update_times.append(time.time() - t0)

    avg_update = sum(update_times) / len(update_times)
    print(f" {NUM_UPDATES} Updates: Avg {avg_update*1000:.2f}ms/update")

    # Scan with multiple patches
    t0 = time.time()
    final_result = store.scan(
        filters=[("region", "=", TARGET_REGION_2)],
        columns=["price", "status", "region"],
    )
    t_multi_patch = time.time() - t0

    # Validation
    final_statuses = final_result["status"].to_numpy()
    correct = np.sum(final_statuses == f"updated_{NUM_UPDATES}")

    print(
        f" Multi-Patch Scan: {len(final_result):,} rows | Time: {t_multi_patch:.4f}s"
    )
    print(
        f"   Validation: {correct}/{multi_count} correct "
        f"({'PASS' if correct == multi_count else 'FAIL'})"
    )

    # Test 9: Baseline (No Patches)
    print("\n--- 9. BASELINE SCAN (No Updates) ---")
    CLEAN_REGION = 9
    t0 = time.time()
    clean_result = store.scan(
        filters=[("region", "=", CLEAN_REGION)], columns=["region", "metric_0", "price"]
    )
    t_baseline = time.time() - t0
    print(f" Baseline: {len(clean_result):,} rows | Time: {t_baseline:.4f}s")
    print(
        f"   MoR Overhead: {t_multi_patch/t_baseline:.2f}x "
        f"(for {NUM_UPDATES} patches)"
    )

    # ==========================================
    # PART 3: ADVANCED SQL FEATURES
    # ==========================================
    print("\n" + "=" * 80)
    print("PART 3: ADVANCED SQL FEATURES")
    print("=" * 80)

    # Test 10: Range Pruning (>=)
    print("\n--- 10. RANGE PRUNING (global_id >= 50000) ---")
    t0 = time.time()
    result = store.scan(
        filters=[("global_id", ">=", TOTAL_ROWS // 2)], columns=["global_id", "region"]
    )
    t_range = time.time() - t0
    print(f" Range Query: {len(result):,} rows | Time: {t_range:.4f}s")
    print(
        f"   ID Range: [{result['global_id'].to_numpy().min()}, "
        f"{result['global_id'].to_numpy().max()}]"
    )

    # Test 11: Range Pruning (<=)
    print("\n--- 11. RANGE PRUNING (price <= 300) ---")
    t0 = time.time()
    result2 = store.scan(filters=[("price", "<=", 300)], columns=["price", "region"])
    t_range2 = time.time() - t0
    print(f" Range Query: {len(result2):,} rows | Time: {t_range2:.4f}s")
    print(f"   Max Price: {result2['price'].to_numpy().max()}")

    # Test 12: Complex Update
    print("\n--- 12. COMPLEX UPDATE (Multi-Condition) ---")
    print(
        "Query: UPDATE SET metric_0 = -99.9 WHERE (region=2 OR region=3) AND price > 800"
    )
    t0 = time.time()
    filters = [
        [("region", "=", 2), "OR", ("region", "=", 3)],
        "AND",
        ("price", ">", 800),
    ]
    store.update(filters, {"metric_0": np.float32(-99.9)})
    t_complex_update = time.time() - t0
    print(f" Complex Update: Time: {t_complex_update*1000:.2f}ms")

    # Verify
    verify = store.scan(
        filters=[("metric_0", "=", np.float32(-99.9))],
        columns=["metric_0", "region", "price"],
    )
    print(f"   Verification: {len(verify):,} rows updated")

    # Cleanup
    store.close()

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_ultimate_benchmark()
