import shutil
import sys
import numpy as np
import pyarrow as pa
from pathlib import Path


from src.core.api import DDIMSession, col


BASE_PATH = "./test_suite_data"


def setup_environment():
    """Cleans and recreates the test data directory."""
    p = Path(BASE_PATH)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def check_result(test_name, computed, expected, tolerance=1e-5):
    """
    Helper function for 'Ground Truth' verification.
    Compares computed results against expected values with floating point tolerance.
    """
    try:
        # Check if inputs are numpy arrays or lists
        if isinstance(computed, (np.ndarray, list)) or isinstance(
            expected, (np.ndarray, list)
        ):
            np.testing.assert_allclose(computed, expected, rtol=tolerance)
        else:
            # Standard scalar comparison
            assert computed == expected
        print(f"  CHECK PASSED: {test_name}")
    except AssertionError:
        print(f"   CHECK FAILED: {test_name}")
        print(f"      Got: {computed}")
        print(f"      Exp: {expected}")
        sys.exit(1)


def run_test_suite():
    setup_environment()
    session = DDIMSession(BASE_PATH)

    # =========================================================================
    # TEST 1: METEO GRID (Slicing + Mean Reduction)
    # =========================================================================
    print("\n" + "=" * 60)
    print(" TEST 1: SCIENTIFIC ARRAY (Meteo Grid)")
    print("=" * 60)

    # 1. Setup Dataset
    df_meteo = session.create_dataset(
        "meteo_grid", hash_dims={"lat": 5}, range_dims={"lon": 100}
    )

    # Generate Synthetic Data
    rows = 30
    temps_hourly = np.random.uniform(10, 35, (rows, 24))
    lats = np.random.randint(40, 45, rows)
    lons = np.arange(rows)

    data_meteo = pa.Table.from_pydict(
        {"lat": lats, "lon": lons, "temperature_profile": temps_hourly.tolist()}
    )
    df_meteo.write(data_meteo)
    print(f"✓ Ingested {rows} rows.")

    # 2. Execute Library Query
    # Logic: Filter Lat 42, Lon [10:20]. Calculate mean of first 6 hours.
    query_meteo = (
        df_meteo[42, 10:20]
        .with_column(
            "morning_avg", col("temperature_profile").v_slice(0, 6).v_reduce("mean")
        )
        .select("lon", "morning_avg")
    )

    # --- VISUAL INSPECTION ---
    query_meteo.explain()
    res_lib = query_meteo.collect().to_pandas().sort_values("lon")
    print("\n--- Result (Head) ---")
    print(res_lib.head())

    # --- GROUND TRUTH VERIFICATION ---
    print("\n Running 'Ground Truth' verification...")

    # Manual Filtering (NumPy)
    mask = (lats == 42) & (lons >= 10) & (lons < 20)

    if np.any(mask):
        filtered_temps = temps_hourly[mask]
        # Manual Calculation: Slice [0:6] -> Mean
        expected_means = np.mean(filtered_temps[:, 0:6], axis=1)

        # Verify Row Count
        check_result("Meteo Row Count", len(res_lib), len(expected_means))
        # Verify Calculated Values
        check_result(
            "Meteo Values (Mean)", res_lib["morning_avg"].values, expected_means
        )
    else:
        print("⚠️  No random data generated for Lat 42 this run. Skipping check.")

    # =========================================================================
    # TEST 2: FINANCE (SQL Filter + Vector Tail)
    # =========================================================================
    print("\n" + "=" * 60)
    print("📈 TEST 2: FINANCIAL DATA (Ticker Lists)")
    print("=" * 60)

    # 1. Setup Dataset
    df_fin = session.create_dataset(
        "market_data", hash_dims={"ticker": 2}, range_dims={"ts": 1000}
    )

    # Generate Synthetic Data
    tickers = ["AAPL", "GOOG", "MSFT"] * 10
    trade_lists = [np.linspace(0, 49, 50).tolist() for _ in range(30)]  # 0..49 per row
    data_fin = pa.Table.from_pydict(
        {"ticker": tickers, "ts": np.arange(30), "trades": trade_lists}
    )
    df_fin.write(data_fin)
    print(f" Ingested 30 rows.")

    # 2. Execute Library Query
    # Logic: Filter 'AAPL', get last 3 trades from the vector
    query_fin = df_fin.filter(col("ticker") == "AAPL").select(
        "ts", col("trades").v_slice(47, 50).alias("last_3")
    )

    # --- VISUAL INSPECTION ---
    query_fin.explain()
    res_fin = query_fin.collect().to_pandas().sort_values("ts")
    print("\n--- Result (Head) ---")
    print(res_fin.head())

    # --- GROUND TRUTH VERIFICATION ---
    print("\n🔎 Running 'Ground Truth' verification...")
    expected_rows = []
    for i, t in enumerate(tickers):
        if t == "AAPL":
            # Manual Slice: last 3 elements
            expected_rows.append(trade_lists[i][47:50])

    check_result("Finance Row Count", len(res_fin), len(expected_rows))
    check_result("Finance Vector Content", res_fin["last_3"].tolist(), expected_rows)

    # =========================================================================
    # TEST 3: AI / VECTOR SEARCH (Cosine + Filter)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: AI / VECTOR SEARCH")
    print("=" * 60)

    # 1. Setup Dataset
    df_ai = session.create_dataset("embeddings", hash_dims={"cat": 2}, global_id="id")

    # Vectors setup:
    # ID 1: Exact Match (1.0)
    # ID 2: Orthogonal (0.0)
    # ID 3: Very Similar (0.9)
    # ID 4: Different (0.1)
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.1, 0.9]]
    ids = [1, 2, 3, 4]

    data_ai = pa.Table.from_pydict({"cat": ["A"] * 4, "id": ids, "vector": vecs})
    df_ai.write(data_ai)
    print(f"✓ Ingested 4 rows.")

    # 2. Execute Library Query
    query_vec = [1.0, 0.0, 0.0]

    # Logic: Calculate Cosine Similarity -> Filter > 0.8 -> Select ID & Score
    query_ai = (
        df_ai.with_column("score", col("vector").cosine_sim(query_vec))
        .filter(col("score") > 0.8)  # This is a MEMORY FILTER
        .select("id", "score")
    )

    # --- VISUAL INSPECTION ---
    query_ai.explain()
    res_ai = query_ai.collect().to_pandas().sort_values("id")
    print("\n--- Result (Matches) ---")
    print(res_ai)

    # --- GROUND TRUTH VERIFICATION ---
    print("\n🔎 Running 'Ground Truth' verification...")
    expected_ids = [1, 3]

    # Verify IDs found
    check_result("AI Found IDs", res_ai["id"].tolist(), expected_ids)

    # Verify Scores (Manual dot product)
    scores_manual = []
    for idx in [0, 2]:  # Indices of ID 1 and ID 3
        v = np.array(vecs[idx])
        q = np.array(query_vec)
        # Cosine Similarity Formula
        scores_manual.append(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q)))

    check_result("AI Similarity Scores", res_ai["score"].values, scores_manual)

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED 'GROUND TRUTH' VALIDATION")
    print("=" * 60)


if __name__ == "__main__":
    run_test_suite()
