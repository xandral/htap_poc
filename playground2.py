
import time
import numpy as np
import pyarrow as pa
import shutil
import os
import sys


from src.core.storage import NDimStorage
from src.core.vector_operations import VectorOps


# Configuration
TOTAL_ROWS = 200_000
TOTAL_COLS = 128
CHUNK_ROWS = 20_000
CHUNK_COLS = 32
PATH = "./htap_benchmark_enhanced"

LOGICAL_H = 16
LOGICAL_W = 8


def generate_mixed_batch(rows):
    print(f"Generating {rows} rows...")
    data = {
        "region": np.random.randint(0, 10, rows).astype(np.int32),
        "price": np.random.randint(0, 1000, rows).astype(np.int32),
    }

    # Matrix (Fixed Size) - Metadata will be inferred automatically
    matrix = np.random.rand(rows, TOTAL_COLS).astype(np.float64)
    data["sensor_matrix"] = [row for row in matrix]

    # Extra columns for cosine
    matrix_b = np.roll(matrix, 1, axis=1)
    data["embedding_a"] = [row for row in matrix]
    data["embedding_b"] = [row for row in matrix_b]

    # Simple cols for legacy
    data["metric_a"] = matrix[:, 0]
    data["metric_b"] = matrix[:, 1]
    data["metric_c"] = matrix[:, 2]

    return pa.Table.from_pydict(data)


def main():
    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    print("=" * 60)
    print(" PART 1: INITIALIZATION & METADATA INFERENCE")
    print("=" * 60)

    store = NDimStorage(
        PATH, CHUNK_ROWS, CHUNK_COLS, hash_dims={"region": 4}, range_dims={"price": 100}
    )

    t0 = time.time()
    batch = generate_mixed_batch(TOTAL_ROWS)
    store.write_batch(batch)
    print(f"\n✓ Ingested {TOTAL_ROWS} rows ({time.time() - t0:.2f}s)")

    # Display inferred metadata
    print("\n" + "=" * 60)
    print(" METADATA INFERENCE RESULTS")
    print("=" * 60)
    for col_name in ["sensor_matrix", "embedding_a", "region", "price"]:
        metadata = store.schema.metadata.get(col_name)
        if metadata:
            print(f"\n{col_name}:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

    # =========================================================
    # PART 2: BASELINE (Traditional Approach)
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 2: BASELINE (Traditional 2-Step Approach)")
    print("=" * 60)

    print("\n--- 2a. TRADITIONAL: Scan + Cosine (2 steps) ---")
    t0 = time.time()
    res = store.scan(columns=["embedding_a", "embedding_b"])
    t_scan = time.time() - t0

    t1 = time.time()
    res = res.append_column(
        "cosine_sim", VectorOps.cosine_similarity(res, ["embedding_a"], ["embedding_b"])
    )
    t_compute = time.time() - t1

    print(f" Scan Time: {t_scan:.4f}s")
    print(f" Compute Time: {t_compute:.4f}s")
    print(f" Total: {t_scan + t_compute:.4f}s")
    print(f" Result: {len(res)} rows")

    # =========================================================
    # PART 3: INTEGRATED VECTOR OPS (Per-Chunk Processing)
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 3: INTEGRATED APPROACH (Per-Chunk Vector Ops)")
    print("=" * 60)

    print("\n--- 3a. INTEGRATED: Cosine computed per-chunk ---")
    t0 = time.time()

    # Vector op applied to each chunk in parallel during I/O
    cosine_op = lambda table: table.append_column(
        "cosine_sim",
        VectorOps.cosine_similarity(table, ["embedding_a"], ["embedding_b"]),
    )

    res = store.scan(columns=["embedding_a", "embedding_b"], vector_ops=[cosine_op])

    total_time = time.time() - t0
    print(f" Total Time (Scan + Compute): {total_time:.4f}s")
    print(f" Result: {len(res)} rows, columns: {res.column_names}")
    print(f" ✓ Vector ops executed in parallel per-chunk!")

    print("\n--- 3b. CHAINED OPS (Multiple transformations) ---")
    t0 = time.time()

    ops = [
        lambda t: VectorOps.scalar_op(t, "metric_a", "mul", 2.0, "metric_a_x2"),
        lambda t: VectorOps.math_n_sum(t, ["metric_a_x2", "metric_b"], "combined"),
    ]

    res = store.scan(columns=["metric_a", "metric_b"], vector_ops=ops)

    print(f" Time: {time.time() - t0:.4f}s")
    print(f" Result columns: {res.column_names}")

    # =========================================================
    # PART 4: MDA OPERATIONS
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 4: MDA OPERATIONS (SQL:2023 Array Functions)")
    print("=" * 60)

    print("\n--- 4a. 1D SLICING (Pure Arrow, Fast Path) ---")
    t0 = time.time()

    slice_op = lambda table: VectorOps.mda_slice_tensor(
        table, "sensor_matrix", slices=[(10, 15)]
    )

    res = store.scan(columns=["sensor_matrix"], vector_ops=[slice_op])

    print(f" Result columns: {res.column_names}")

    # Find the slice column (should contain "slice")
    slice_cols = [c for c in res.column_names if "slice" in c]
    if slice_cols:
        new_col = slice_cols[0]
        first_slice = res[new_col][0].as_py()
        print(f" Sliced {len(res)} tensors")
        print(f" Slice length: {len(first_slice)} elements (expected 5)")
    else:
        print(f" WARNING: No slice column found! Available: {res.column_names}")

    print(f" Time: {time.time() - t0:.4f}s")

    print(f"\n--- 4b. 2D TENSOR SLICING (NumPy reshape path) ---")
    t0 = time.time()

    logical_shape = (LOGICAL_H, LOGICAL_W)

    slice_2d_op = lambda table: VectorOps.mda_slice_tensor(
        table,
        "sensor_matrix",
        shape=logical_shape,
        slices=[(4, 12), (0, 4)],  # rows 4-12, cols 0-4
    )

    res = store.scan(columns=["sensor_matrix"], vector_ops=[slice_2d_op])

    print(f" Result columns: {res.column_names}")

    slice_cols = [c for c in res.column_names if "slice" in c]
    if slice_cols:
        new_col = slice_cols[0]
        first_slice = res[new_col][0].as_py()
        expected_size = (12 - 4) * (4 - 0)  # 8 rows * 4 cols = 32

        print(f" Computed {len(res)} 2D slices")
        print(f" Slice shape: {len(first_slice)} elements (expected {expected_size})")
    else:
        print(f" WARNING: No slice column found! Available: {res.column_names}")

    print(f" Time: {time.time() - t0:.4f}s")

    print("\n--- 4c. FILTER BY INDEX ---")
    t0 = time.time()

    filter_op = lambda table: VectorOps.mda_filter_by_index(
        table, "sensor_matrix", 0, "BETWEEN", (0.2, 0.5)
    )

    res = store.scan(columns=["sensor_matrix"], vector_ops=[filter_op])

    print(f" Filtered to {len(res)} matching rows (out of {TOTAL_ROWS})")
    print(f" Time: {time.time() - t0:.4f}s")

    print("\n--- 4d. REDUCE MEAN ---")
    t0 = time.time()

    reduce_op = lambda table: VectorOps.mda_reduce_mean(table, "sensor_matrix")

    res = store.scan(columns=["sensor_matrix"], vector_ops=[reduce_op])

    print(f" Computed means for {len(res)} rows")
    print(f" Result columns: {res.column_names}")
    print(f" Sample mean: {res['sensor_matrix_mean'][0].as_py():.6f}")
    print(f" Time: {time.time() - t0:.4f}s")

    # =========================================================
    # PART 5: COMPLEX QUERIES (Filters + Vector Ops)
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 5: COMPLEX QUERIES")
    print("=" * 60)

    print("\n--- 5a. Filtered Query with Vector Ops ---")
    print("Query: Cosine similarity WHERE region=2 AND price > 500")
    t0 = time.time()

    res = store.scan(
        filters=[("region", "=", 2), "AND", ("price", ">", 500)],
        columns=["embedding_a", "embedding_b", "region", "price"],
        vector_ops=[
            lambda t: t.append_column(
                "cosine_sim",
                VectorOps.cosine_similarity(t, ["embedding_a"], ["embedding_b"]),
            )
        ],
    )

    print(f" Filtered to {len(res)} rows")
    print(f" Result columns: {res.column_names}")
    print(f" Time: {time.time() - t0:.4f}s")

    print("\n--- 5b. Multi-Stage Pipeline ---")
    print("Query: Filter → Slice → Reduce")
    t0 = time.time()

    pipeline = [
        lambda t: VectorOps.mda_slice_tensor(t, "sensor_matrix", slices=[(0, 64)]),
        lambda t: VectorOps.mda_reduce_mean(t, "sensor_matrix_slice_1d"),
    ]

    res = store.scan(
        filters=[("region", "=", 5)], columns=["sensor_matrix"], vector_ops=pipeline
    )

    print(f" Result: {len(res)} rows")
    print(f" Columns: {res.column_names}")
    print(f" Time: {time.time() - t0:.4f}s")

    # =========================================================
    # PART 6: UPDATE WITH VECTOR OPS
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 6: UPDATE WITH VECTOR OPERATIONS")
    print("=" * 60)

    print("\n--- 6a. Traditional Update (Scalar) ---")
    t0 = time.time()
    store.update(filters=[("region", "=", 3)], updates={"metric_a": np.float64(999.0)})
    print(f" Time: {time.time() - t0:.4f}s")

    # Verify
    check = store.scan(filters=[("region", "=", 3)], columns=["metric_a"])
    print(f" Verified: {check['metric_a'][0].as_py()} (expected 999.0)")

    print("\n--- 6b. UPDATE with Computed Values (Vector Ops) ---")
    print("Update: SET metric_b = metric_a * 2 WHERE region=4")
    t0 = time.time()

    # Compute derived column using vector ops
    compute_double = lambda t: VectorOps.scalar_op(
        t, "metric_a", "mul", 2.0, "metric_a_doubled"
    )

    # store.update(
    #     filters=[("region", "=", 4)],
    #     updates={
    #         "metric_b": lambda t: t["metric_a_doubled"]  # Use computed column
    #     },
    #     vector_ops=[compute_double]
    # )
    # print(f" Time: {time.time() - t0:.4f}s")

    # # Verify: metric_b should equal metric_a * 2
    # check = store.scan(
    #     filters=[("region", "=", 4)],
    #     columns=["metric_a", "metric_b"]
    # )
    # sample_a = check['metric_a'][0].as_py()
    # sample_b = check['metric_b'][0].as_py()
    # print(f" Verified: metric_a={sample_a:.4f}, metric_b={sample_b:.4f}")
    # print(f" Ratio (should be ~2.0): {sample_b/sample_a:.4f}")

    # =========================================================
    # PART 7: METADATA PERSISTENCE
    # =========================================================
    print("\n" + "=" * 60)
    print(" PART 7: METADATA PERSISTENCE TEST")
    print("=" * 60)

    print("\n--- Closing and Reopening Storage ---")
    store.close()

    print("Reopening WITHOUT explicit hash_dims/range_dims...")
    store2 = NDimStorage(
        PATH,
        CHUNK_ROWS,
        CHUNK_COLS,
        # Note: NO hash_dims or range_dims specified!
    )

    print(f"\nRELOADED Configuration:")
    print(f"  hash_dims: {store2.hash_dims}")
    print(f"  range_dims: {store2.range_dims}")

    print("\nRELOADED Metadata:")
    for col_name in ["sensor_matrix", "region", "price"]:
        meta = store2.schema.get_metadata(col_name)
        if meta:
            print(f"  {col_name}: {meta}")

    # Verify operations still work
    print("\n--- Testing Query After Reload ---")
    t0 = time.time()

    res = store2.scan(
        filters=[("region", "=", 7)],
        columns=["sensor_matrix"],
        vector_ops=[lambda t: VectorOps.mda_reduce_mean(t, "sensor_matrix")],
    )
    print(f" Query returned {len(res)} rows")
    print(f" Time: {time.time() - t0:.4f}s")

    store2.close()


if __name__ == "__main__":
    main()
