"""
Benchmark: NDimStorage vs DuckDB
Focus: Coordinate-based pruning on IoT sensor data

Setup:
- 1M rows, timestamps 1 → 1,000,000
- 5 sensors (hash dimension)
- 128 numeric columns (IoT payload)
"""

import time
import shutil
import tempfile
import statistics
from pathlib import Path
from typing import Callable, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import duckdb

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.storage import NDimStorage
from src.core.api import DDIMSession, col


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "num_rows": 1_000_000,
    "num_sensors": 5,
    "num_data_columns": 126,  # + timestamp + sensor_id = 128 total
    "chunk_rows": 10_000,
    "chunk_cols": 32,
    "hash_buckets": 5,
    "warmup_runs": 2,
    "benchmark_runs": 5,
}

SENSORS = [f"sensor_{chr(65 + i)}" for i in range(CONFIG["num_sensors"])]


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_iot_data(num_rows: int, seed: int = 42) -> pa.Table:
    """Generate synthetic IoT sensor data."""
    print(f"\n{'='*60}")
    print(f"Generating {num_rows:,} rows of IoT data...")
    print(f"{'='*60}")

    np.random.seed(seed)
    t0 = time.time()

    # Primary dimensions
    timestamps = np.arange(1, num_rows + 1, dtype=np.int64)
    sensors = np.array([SENSORS[i % len(SENSORS)] for i in range(num_rows)])

    # Data columns (simulate sensor readings)
    data = {
        "timestamp": timestamps,
        "sensor_id": sensors,
    }

    # Generate 126 numeric columns (temp, humidity, pressure, etc.)
    for i in range(CONFIG["num_data_columns"]):
        col_name = f"metric_{i:03d}"
        if i % 3 == 0:
            # Integer metrics (counts, status codes)
            data[col_name] = np.random.randint(0, 1000, size=num_rows)
        elif i % 3 == 1:
            # Float metrics (temperatures, percentages)
            data[col_name] = np.random.uniform(0, 100, size=num_rows).astype(np.float32)
        else:
            # Large float metrics (energy readings)
            data[col_name] = np.random.uniform(0, 10000, size=num_rows).astype(np.float64)

    table = pa.Table.from_pydict(data)

    elapsed = time.time() - t0
    size_mb = sum(col.nbytes for col in table.columns) / (1024 * 1024)

    print(f"  Rows: {num_rows:,}")
    print(f"  Columns: {len(table.column_names)}")
    print(f"  Size in memory: {size_mb:.1f} MB")
    print(f"  Generation time: {elapsed:.2f}s")

    return table


# =============================================================================
# STORAGE SETUP
# =============================================================================

def setup_ndim_storage(data: pa.Table, base_path: Path) -> NDimStorage:
    """Initialize and populate NDimStorage."""
    print(f"\n{'='*60}")
    print("Setting up NDimStorage...")
    print(f"{'='*60}")

    # Clean previous data
    base_path = Path(base_path)
    if base_path.exists():
        shutil.rmtree(base_path)

    t0 = time.time()

    storage = NDimStorage(
        str(base_path),
        chunk_rows=CONFIG["chunk_rows"],
        chunk_cols=CONFIG["chunk_cols"],
        hash_dims={"sensor_id": CONFIG["hash_buckets"]},
        range_dims={"timestamp": CONFIG["chunk_rows"]},  # Range partition by timestamp
        global_id="timestamp",  # Use timestamp as global ID
    )

    # Write data
    storage.write_batch(data)

    elapsed = time.time() - t0

    # Count files
    # parquet_files = list(base_path.glob("*.parquet"))
    # total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)

    print(f"  Chunk config: rows={CONFIG['chunk_rows']:,}, cols={CONFIG['chunk_cols']}")
    print(f"  Hash dimension: sensor_id ({CONFIG['hash_buckets']} buckets)")
    print(f"  Range dimension: timestamp (interval={CONFIG['chunk_rows']:,})")
    # print(f"  Parquet files created: {len(parquet_files)}")
    # print(f"  Total storage size: {total_size:.1f} MB")
    print(f"  Ingestion time: {elapsed:.2f}s")

    return storage


def setup_duckdb(data: pa.Table, db_path: Path) -> duckdb.DuckDBPyConnection:
    """Initialize and populate DuckDB."""
    print(f"\n{'='*60}")
    print("Setting up DuckDB...")
    print(f"{'='*60}")

    # Clean previous
    if db_path.exists():
        db_path.unlink()

    t0 = time.time()

    conn = duckdb.connect(str(db_path))

    # Create table from Arrow
    conn.execute("CREATE TABLE iot_data AS SELECT * FROM data")

    # Create indexes for fair comparison
    conn.execute("CREATE INDEX idx_timestamp ON iot_data(timestamp)")
    conn.execute("CREATE INDEX idx_sensor ON iot_data(sensor_id)")
    conn.execute("CREATE INDEX idx_combined ON iot_data(timestamp, sensor_id)")

    elapsed = time.time() - t0

    # Get stats
    row_count = conn.execute("SELECT COUNT(*) FROM iot_data").fetchone()[0]
    db_size = db_path.stat().st_size / (1024 * 1024)

    print(f"  Rows loaded: {row_count:,}")
    print(f"  Database size: {db_size:.1f} MB")
    print(f"  Indexes created: timestamp, sensor_id, combined")
    print(f"  Setup time: {elapsed:.2f}s")

    return conn


# =============================================================================
# BENCHMARK QUERIES
# =============================================================================

@dataclass
class BenchmarkQuery:
    name: str
    description: str
    selectivity: str
    ndim_query: Callable[[NDimStorage], pa.Table]
    duckdb_query: str
    ndim_filters: List = None  # For candidate count


def count_candidates(storage: NDimStorage, filters: List) -> int:
    """Count how many files would be read after pruning."""
    _, candidates = storage.scan(filters=filters, return_candidates=True, columns=["timestamp"])
    return len(candidates)


def define_queries() -> List[BenchmarkQuery]:
    """Define benchmark queries from most to least selective."""

    # Target values for queries
    # Sensor assignment: SENSORS[i % 5] where i is 0-indexed
    # sensor_E: i % 5 == 4 -> timestamps 5,10,15,... 500000
    exact_ts = 500_000  # belongs to sensor_E
    range_start = 100_000
    range_end_narrow = 100_100  # 100 rows
    range_end_medium = 200_000  # 100k rows
    target_sensor = "sensor_E"  # matches timestamps divisible by 5

    queries = [
        # Q0: Timestamp-only point lookup (row-chunk pruning demo)
        # NDim: calculates row_chunk = 500000 // 100000 = 5, reads only r5 chunks
        # Without pruning would read all 164 files, with pruning reads ~4 files (1 row chunk × 4 col chunks)
        BenchmarkQuery(
            name="Q0_timestamp_only",
            description=f"timestamp={exact_ts} (row-chunk pruning)",
            selectivity="~1 row, ~4 files vs 164",
            ndim_query=lambda s: s.scan(
                filters=[("timestamp", "=", exact_ts)],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE timestamp = {exact_ts}
            """
        ),

        # Q1: Point lookup (maximum pruning: row + hash)
        BenchmarkQuery(
            name="Q1_point_lookup",
            description=f"timestamp={exact_ts} AND sensor='{target_sensor}'",
            selectivity="~1 row, 1 file",
            ndim_query=lambda s: s.scan(
                filters=[
                    ("timestamp", "=", exact_ts),
                    ("sensor_id", "=", target_sensor)
                ],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE timestamp = {exact_ts} AND sensor_id = '{target_sensor}'
            """
        ),

        # Q2: Narrow range + hash filter
        BenchmarkQuery(
            name="Q2_narrow_range_hash",
            description=f"timestamp [{range_start}-{range_end_narrow}] AND sensor='{target_sensor}'",
            selectivity="~20 rows",
            ndim_query=lambda s: s.scan(
                filters=[
                    [("timestamp", ">=", range_start), "AND", ("timestamp", "<", range_end_narrow)],
                    "AND",
                    ("sensor_id", "=", target_sensor)
                ],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE timestamp >= {range_start} AND timestamp < {range_end_narrow}
                  AND sensor_id = '{target_sensor}'
            """
        ),

        # Q3: Medium range + hash filter
        BenchmarkQuery(
            name="Q3_medium_range_hash",
            description=f"timestamp [{range_start}-{range_end_medium}] AND sensor='{target_sensor}'",
            selectivity="~20k rows",
            ndim_query=lambda s: s.scan(
                filters=[
                    [("timestamp", ">=", range_start), "AND", ("timestamp", "<", range_end_medium)],
                    "AND",
                    ("sensor_id", "=", target_sensor)
                ],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE timestamp >= {range_start} AND timestamp < {range_end_medium}
                  AND sensor_id = '{target_sensor}'
            """
        ),

        # Q4: Range filter only (no hash)
        BenchmarkQuery(
            name="Q4_range_only",
            description=f"timestamp [{range_start}-{range_end_medium}]",
            selectivity="~100k rows",
            ndim_query=lambda s: s.scan(
                filters=[
                    ("timestamp", ">=", range_start),
                    "AND",
                    ("timestamp", "<", range_end_medium)
                ],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE timestamp >= {range_start} AND timestamp < {range_end_medium}
            """
        ),

        # Q5: Hash filter only (all timestamps for one sensor)
        BenchmarkQuery(
            name="Q5_hash_only",
            description=f"sensor='{target_sensor}' (all timestamps)",
            selectivity="~200k rows",
            ndim_query=lambda s: s.scan(
                filters=[("sensor_id", "=", target_sensor)],
                columns=["timestamp", "sensor_id", "metric_000", "metric_001"]
            ),
            duckdb_query=f"""
                SELECT timestamp, sensor_id, metric_000, metric_001
                FROM iot_data
                WHERE sensor_id = '{target_sensor}'
            """
        ),
    ]

    return queries


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

@dataclass
class QueryResult:
    query_name: str
    system: str
    times_ms: List[float]
    row_count: int

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0


def show_pruning_demo(ndim_storage: NDimStorage, total_files: int):
    """Demonstrate coordinate-based pruning effectiveness."""
    print(f"\n{'='*60}")
    print("PRUNING DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Total parquet files: {total_files}")
    print()

    test_cases = [
        ("No filter (full scan)", []),
        ("timestamp=500000 (row pruning)", [("timestamp", "=", 500_000)]),
        ("sensor='sensor_E' (hash pruning)", [("sensor_id", "=", "sensor_E")]),
        ("timestamp=500000 AND sensor='sensor_E' (row+hash)", [
            ("timestamp", "=", 500_000),
            ("sensor_id", "=", "sensor_E")
        ]),
    ]

    print(f"{'Filter':<45} {'Files Read':<12} {'Reduction':<10}")
    print("-" * 70)

    for desc, filters in test_cases:
        _, candidates = ndim_storage.scan(
            filters=filters,
            return_candidates=True,
            columns=["timestamp"]
        )
        n_files = len(candidates)
        reduction = f"{(1 - n_files/total_files)*100:.0f}%" if total_files > 0 else "N/A"
        print(f"{desc:<45} {n_files:<12} {reduction:<10}")

    print()


def run_benchmark(
    queries: List[BenchmarkQuery],
    ndim_storage: NDimStorage,
    duckdb_conn: duckdb.DuckDBPyConnection,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> Dict[str, List[QueryResult]]:
    """Run all benchmark queries."""

    print(f"\n{'='*60}")
    print("Running Benchmark")
    print(f"{'='*60}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Benchmark runs: {benchmark_runs}")

    results = {}

    for q in queries:
        print(f"\n--- {q.name}: {q.description} ---")
        print(f"    Expected selectivity: {q.selectivity}")

        # Warmup NDim
        for _ in range(warmup_runs):
            _ = q.ndim_query(ndim_storage)

        # Warmup DuckDB
        for _ in range(warmup_runs):
            _ = duckdb_conn.execute(q.duckdb_query).fetch_arrow_table()

        # Benchmark NDim
        ndim_times = []
        ndim_rows = 0
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            result = q.ndim_query(ndim_storage)
            elapsed = (time.perf_counter() - t0) * 1000
            ndim_times.append(elapsed)
            ndim_rows = len(result)

        # Benchmark DuckDB (use fetch_arrow_table() for fair comparison with PyArrow Table)
        duck_times = []
        duck_rows = 0
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            result = duckdb_conn.execute(q.duckdb_query).fetch_arrow_table()
            elapsed = (time.perf_counter() - t0) * 1000
            duck_times.append(elapsed)
            duck_rows = len(result)

        # Store results
        ndim_result = QueryResult(q.name, "NDimStorage", ndim_times, ndim_rows)
        duck_result = QueryResult(q.name, "DuckDB", duck_times, duck_rows)

        results[q.name] = [ndim_result, duck_result]

        # Print comparison
        if ndim_result.mean_ms > 0 and duck_result.mean_ms > 0:
            if ndim_result.mean_ms < duck_result.mean_ms:
                winner = "NDim"
                ratio = duck_result.mean_ms / ndim_result.mean_ms
            else:
                winner = "DuckDB"
                ratio = ndim_result.mean_ms / duck_result.mean_ms
        else:
            winner = "N/A"
            ratio = 0

        print(f"    NDim:   {ndim_result.mean_ms:8.2f} ms (±{ndim_result.std_ms:.2f}) | {ndim_rows:,} rows")
        print(f"    DuckDB: {duck_result.mean_ms:8.2f} ms (±{duck_result.std_ms:.2f}) | {duck_rows:,} rows")
        print(f"    Winner: {winner} ({ratio:.2f}x faster)" if ratio > 1.01 else "    Winner: ~TIE")

    return results


# =============================================================================
# REPORT
# =============================================================================

def print_report(results: Dict[str, List[QueryResult]]):
    """Print final benchmark report."""

    print(f"\n{'='*80}")
    print("BENCHMARK REPORT: NDimStorage vs DuckDB")
    print(f"{'='*80}")

    print(f"\n{'Query':<25} {'NDim (ms)':<15} {'DuckDB (ms)':<15} {'Ratio':<12} {'Winner':<10}")
    print("-" * 80)

    ndim_wins = 0
    duck_wins = 0

    for name, (ndim, duck) in results.items():
        if ndim.mean_ms > 0 and duck.mean_ms > 0:
            if ndim.mean_ms < duck.mean_ms:
                # NDim is faster
                ratio = duck.mean_ms / ndim.mean_ms
                if ratio > 1.1:
                    winner = "NDim"
                    ndim_wins += 1
                    ratio_str = f"NDim {ratio:.2f}x"
                else:
                    winner = "~TIE"
                    ratio_str = "~1.00x"
            else:
                # DuckDB is faster
                ratio = ndim.mean_ms / duck.mean_ms
                if ratio > 1.1:
                    winner = "DuckDB"
                    duck_wins += 1
                    ratio_str = f"Duck {ratio:.2f}x"
                else:
                    winner = "~TIE"
                    ratio_str = "~1.00x"
        else:
            winner = "N/A"
            ratio_str = "N/A"

        print(f"{name:<25} {ndim.mean_ms:<15.2f} {duck.mean_ms:<15.2f} {ratio_str:<12} {winner:<10}")

    print("-" * 80)
    print(f"\nSummary: NDim wins {ndim_wins}, DuckDB wins {duck_wins}, Ties {len(results) - ndim_wins - duck_wins}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    print("""
NDimStorage advantages:
  - Coordinate-based pruning (hash + range dimensions)
  - Direct file lookup without index traversal
  - Efficient for selective queries on partitioned data

DuckDB advantages:
  - Highly optimized vectorized scan engine
  - Better parallelism (no GIL)
  - Mature query optimizer with statistics

Query analysis:
  - Q0-Q1: Point lookups - test row/hash pruning
  - Q2-Q3: Range + hash - test combined dimension pruning
  - Q4: Range only - test range dimension effectiveness
  - Q5: Hash only - test hash dimension effectiveness
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main benchmark entry point."""

    print("\n" + "=" * 80)
    print("  NDimStorage vs DuckDB Benchmark - IoT Sensor Data")
    print("=" * 80)

    # Setup paths
    tmp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
    ndim_path = tmp_dir / "ndim_storage"
    duck_path = tmp_dir / "duckdb.db"

    print(f"\nTemp directory: {tmp_dir}")

    try:
        # 1. Generate data
        data = generate_iot_data(CONFIG["num_rows"])

        # 2. Setup storage systems
        ndim_storage = setup_ndim_storage(data, ndim_path)
        duckdb_conn = setup_duckdb(data, duck_path)

        # 3. Define queries
        queries = define_queries()

        # 3.5 Show pruning effectiveness
        total_files = len(list(ndim_path.glob("**/*.parquet")))
        show_pruning_demo(ndim_storage, total_files)

        # 4. Run benchmark
        results = run_benchmark(
            queries,
            ndim_storage,
            duckdb_conn,
            warmup_runs=CONFIG["warmup_runs"],
            benchmark_runs=CONFIG["benchmark_runs"],
        )

        # 5. Print report
        print_report(results)

        # Cleanup
        ndim_storage.close()
        duckdb_conn.close()

    finally:
        # Optional: keep temp files for inspection
        print(f"\nTemp files at: {tmp_dir}")
        print("(Delete manually or uncomment cleanup)")
        # shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
