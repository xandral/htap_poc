# Partition Pruning Benchmark: Rust vs DuckDB

Quantitative evaluation of coordinate-based partition pruning against an in-memory analytical database.

**Note**: Results presented are from single-run benchmarks on a specific hardware configuration. Variance across runs and environments is expected. Use these results as directional guidance, not absolute performance guarantees.

## Hypothesis

Partition pruning on disk-resident data can achieve competitive query latency against in-memory databases when filters eliminate a sufficient fraction of partitions. This trade-off favors pruning-based approaches at scale, where in-memory solutions become impractical.

## Experimental Setup

### Data Generation

Synthetic IoT sensor data with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rows` | 5,000,000 | Total records |
| `num_sensors` | 5 | Hash partition buckets |
| `chunk_rows` | 10,000 | Rows per partition |
| `chunk_cols` | 4 | Column groups per partition |
| `num_data_columns` | 128 | Metric columns |

Data schema:
- `timestamp`: INT64, monotonic, range [1, num_rows]
- `sensor_id`: STRING, cyclic assignment (sensor_A, sensor_B, ...)
- `metric_000..N`: INT64/FLOAT32/FLOAT64

### Data Pipeline

1. Generate data in batches (500K rows/batch)
2. Insert into DuckDB table (ground truth)
3. Export chunked parquet files for Rust scanner

This ensures both systems operate on identical data.

### Partitioning Strategy

Files follow the naming convention:
```
chunk_r{row_chunk}_c{col_chunk}_h{hash_bucket}_r{range}_v{version}.parquet
```

Coordinates:
- **Row chunk**: `timestamp / chunk_rows`
- **Hash bucket**: `SHA256(sensor_id) % num_sensors`
- **Column chunk**: Horizontal partitioning of metric columns
- **Version**: Copy-on-Write versioning for HTAP workloads

### Systems Under Test

| System | Implementation | Data Location | Memory Model |
|--------|---------------|---------------|--------------|
| Rust | Custom Parquet reader with coordinate-based pruning | Disk | Streaming |
| DuckDB | Persistent database file | Disk + Cache | Lazy loading |

### Query Workload

| Query | Filter | Pruning Type |
|-------|--------|--------------|
| Q1 | `timestamp = X` | Row chunk |
| Q2 | `sensor_id = Y` | Hash bucket |
| Q3 | `timestamp = X AND sensor_id = Y` | Row + Hash |
| Q4 | `sensor_id = Y` (full scan baseline) | None |
| Q5 | `timestamp BETWEEN A AND B AND sensor_id = Y` | Range + Hash |
| Q6 | Wide/narrow range variants | Variable |

## Results

### Test Configuration: 5M rows, 128 columns

- 5,000,000 rows
- 2,000 parquet files (col_chunk 0 only for scanning)
- 5 sensors (hash buckets)
- 10,000 rows per chunk
- 128 metric columns
- Disk size: 3.8 GB

#### System Resources

| System | Storage | Load Time |
|--------|---------|-----------|
| Rust | Streams from disk | 0 s |
| DuckDB | 14 MB resident | 0.01 s |

#### Query Latency (p50, milliseconds)

| Query | Pruning % | Rust | DuckDB | Winner |
|-------|-----------|------|--------|--------|
| Q1: timestamp = X | 99.8% | 0.15 | 0.81 | Rust 5.3x |
| Q2: sensor_id = Y | 75.0% | 12.12 | 3.03 | DuckDB 4.0x |
| Q3: timestamp AND sensor | 100.0% | 0.12 | 1.15 | Rust 9.3x |
| Q4: full scan | 0.0% | 64.86 | 3.45 | DuckDB 18.8x |
| Q5: range + sensor | 98.4% | 1.45 | 1.88 | Rust 1.3x |
| Q6: wide range + sensor | 87.5% | 8.91 | 2.80 | DuckDB 3.2x |
| Q7: small range + sensor | 99.7% | 0.95 | 1.75 | Rust 1.8x |

#### Summary

- **Rust wins**: 4 queries (high pruning > 98%)
- **DuckDB wins**: 3 queries (low/moderate pruning < 90%)
- **Breakeven**: ~98% pruning

### Pruning Efficiency Threshold

| Pruning Level | Winner | Typical Speedup |
|---------------|--------|-----------------|
| > 99% | Rust | 5-9x |
| 98-99% | Rust | 1.3-1.8x |
| 87-98% | DuckDB | 3-4x |
| < 75% | DuckDB | 4-19x |

## Analysis

### When Partition Pruning Wins

1. **Point queries on partitioned columns**: Q3 achieves 9.3x speedup by reading 1 file instead of 2,000
2. **High selectivity filters**: Queries touching < 2% of data benefit from I/O elimination
3. **Memory-constrained environments**: Rust processes arbitrarily large datasets with constant memory

### When In-Memory/Cached Wins

1. **Low selectivity filters**: Q4 (full scan) shows 18.8x DuckDB advantage
2. **Moderate pruning (75-90%)**: Disk I/O overhead exceeds cached scan cost
3. **Repeated queries**: DuckDB benefits from OS page cache after first query

### Scalability Projection

At 1 billion rows (extrapolated):

| Metric | Rust | DuckDB |
|--------|------|--------|
| Memory | ~0 MB | ~3 GB+ |
| Point query (Q3) | <1 ms | ~1 ms |
| Full scan (Q4) | ~13 s | ~700 ms |

Rust maintains constant memory footprint regardless of data size.

## Reproduction

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python dependencies
cd scripts/rust_benchmark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Benchmark

```bash
# Standard configuration (5M rows, 128 columns)
python run_benchmark.py --rows 5000000 --data-columns 128 --name "5M_128cols"

# Custom configuration
python run_benchmark.py --rows 1000000 --chunk-rows 10000 --sensors 5 --data-columns 40

# Parameter sweep
python run_benchmark.py --sweep
```

### Output

- `benchmark_data/`: Generated parquet files and DuckDB database
- `results/result_*.json`: Raw timing data
- Console output: Summary statistics and conclusions

## Implementation Details

### Rust Pruning Logic

1. Parse filename to extract coordinates (row_chunk, col_chunk, hash_bucket)
2. Filter to col_chunk=0 files only (avoid duplicate counting)
3. Compute target coordinates from filter predicates
4. Select files matching target coordinates
5. For each selected file:
   - Read Parquet metadata (row group statistics)
   - Prune row groups using min/max statistics
   - Apply column projection
   - Scan with vectorized filters (Arrow compute kernels)

### DuckDB Baseline

```sql
-- Data loaded from persistent database file
-- Table created during data generation phase
SELECT COUNT(*) FROM iot WHERE <filter>;
```

DuckDB uses its native storage format with lazy loading and OS page cache.

## Limitations

1. **Single-run tests**: Results are from individual benchmark executions. Statistical variance is captured via p50/p95 within each run (15 iterations after 5 warmup), but cross-run variance is not measured.
2. **Synthetic data**: Real workloads may have different access patterns and data distributions.
3. **Single-node**: No distributed execution comparison.
4. **Warm cache effects**: After first query, OS page cache benefits both systems differently.
5. **Count-only queries**: Only COUNT(*) measured. Projections, aggregations, and joins not benchmarked.
6. **Hardware-specific**: Results depend on CPU, disk I/O speed, and memory bandwidth of test machine.

## References

- Parquet format specification: https://parquet.apache.org/docs/
- DuckDB: https://duckdb.org/
- Arrow Rust implementation: https://arrow.apache.org/rust/
