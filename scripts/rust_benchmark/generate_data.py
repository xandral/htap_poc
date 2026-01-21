#!/usr/bin/env python3
"""
Parametric Data Generator for Partition Pruning Benchmark.

Flow:
1. Generate data in batches, insert into DuckDB table
2. Export chunked parquet files for Rust (row_chunk x col_chunk x hash_bucket)

Output format: chunk_r{row}_c{col}_h{hash}_r{range}_v{version}.parquet
"""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 500_000  # Insert 500K rows at a time into DuckDB


@dataclass
class DataConfig:
    """Configuration for data generation."""
    num_rows: int = 1_000_000
    num_sensors: int = 5
    num_data_columns: int = 10
    chunk_rows: int = 10_000
    chunk_cols: int = 4  # Number of column groups for Rust
    seed: int = 42
    output_dir: str = "./benchmark_data"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> 'DataConfig':
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_hash_bucket(value: str, num_buckets: int) -> int:
    """SHA256-based hash bucket (must match Rust implementation)."""
    h = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(h[:8], 'big') % num_buckets


def generate_sensors(num_sensors: int) -> List[str]:
    """Generate sensor names."""
    return [f"sensor_{chr(65 + i)}" for i in range(num_sensors)]


def generate_batch(start_row: int, end_row: int, config: DataConfig) -> pa.Table:
    """Generate a batch of rows."""
    batch_size = end_row - start_row
    sensors = generate_sensors(config.num_sensors)

    timestamps = np.arange(start_row + 1, end_row + 1, dtype=np.int64)
    sensor_ids = np.array([sensors[i % len(sensors)] for i in range(start_row, end_row)])

    data = {
        "timestamp": timestamps,
        "sensor_id": sensor_ids,
    }

    np.random.seed(config.seed + start_row)
    for i in range(config.num_data_columns):
        col_name = f"metric_{i:03d}"
        if i % 3 == 0:
            data[col_name] = np.random.randint(0, 1000, size=batch_size, dtype=np.int64)
        elif i % 3 == 1:
            data[col_name] = np.random.uniform(0, 100, size=batch_size).astype(np.float32)
        else:
            data[col_name] = np.random.uniform(0, 10000, size=batch_size).astype(np.float64)

    return pa.Table.from_pydict(data)


def create_duckdb_table(conn: duckdb.DuckDBPyConnection, config: DataConfig):
    """Create the iot table schema in DuckDB."""
    cols = ["timestamp BIGINT", "sensor_id VARCHAR"]
    for i in range(config.num_data_columns):
        col_name = f"metric_{i:03d}"
        if i % 3 == 0:
            cols.append(f"{col_name} BIGINT")
        elif i % 3 == 1:
            cols.append(f"{col_name} FLOAT")
        else:
            cols.append(f"{col_name} DOUBLE")

    schema = ", ".join(cols)
    conn.execute(f"CREATE TABLE iot ({schema})")


def populate_duckdb(conn: duckdb.DuckDBPyConnection, config: DataConfig, batch_size: int):
    """Insert data into DuckDB in batches."""
    num_batches = (config.num_rows + batch_size - 1) // batch_size

    print(f"Inserting {config.num_rows:,} rows in {num_batches} batches...")

    for i in range(num_batches):
        start_row = i * batch_size
        end_row = min((i + 1) * batch_size, config.num_rows)

        batch = generate_batch(start_row, end_row, config)
        conn.execute("INSERT INTO iot SELECT * FROM batch")

        print(f"  Batch {i+1}/{num_batches}: rows {start_row+1:,} - {end_row:,}")


def chunk_and_write(conn: duckdb.DuckDBPyConnection, config: DataConfig) -> tuple:
    """
    Export data from DuckDB as chunked parquet files for Rust.
    Partitioned by (row_chunk, col_chunk, hash_bucket).
    """
    output_dir = Path(config.output_dir)
    sensors = generate_sensors(config.num_sensors)

    # Get all column names
    all_columns = [desc[0] for desc in conn.execute("DESCRIBE iot").fetchall()]
    data_columns = [c for c in all_columns if c not in ('timestamp', 'sensor_id')]

    # Split columns into groups
    cols_per_group = max(1, len(data_columns) // config.chunk_cols)
    column_groups = []
    for i in range(config.chunk_cols):
        start = i * cols_per_group
        end = start + cols_per_group if i < config.chunk_cols - 1 else len(data_columns)
        group_cols = ['timestamp', 'sensor_id'] + data_columns[start:end]
        column_groups.append(group_cols)

    num_row_chunks = (config.num_rows + config.chunk_rows - 1) // config.chunk_rows

    print(f"Exporting: {num_row_chunks} row chunks x {len(column_groups)} col groups x {config.num_sensors} hash buckets")

    files_written = 0
    total_bytes = 0

    for row_chunk in range(num_row_chunks):
        ts_start = row_chunk * config.chunk_rows + 1
        ts_end = min((row_chunk + 1) * config.chunk_rows, config.num_rows)

        for col_idx, col_group in enumerate(column_groups):
            cols_str = ", ".join(col_group)

            for sensor in sensors:
                hash_bucket = get_hash_bucket(sensor, config.num_sensors)

                query = f"""
                    SELECT {cols_str} FROM iot
                    WHERE timestamp >= {ts_start}
                      AND timestamp <= {ts_end}
                      AND sensor_id = '{sensor}'
                """
                result = conn.execute(query).fetch_arrow_table()

                if result.num_rows == 0:
                    continue

                filename = f"chunk_r{row_chunk}_c{col_idx}_h{hash_bucket}_r{row_chunk}_v1.parquet"
                filepath = output_dir / filename
                pq.write_table(result, filepath, compression='snappy')

                files_written += 1
                total_bytes += filepath.stat().st_size

    return files_written, total_bytes


def export_hive_partitioned(conn: duckdb.DuckDBPyConnection, config: DataConfig) -> int:
    """
    Export data in Hive-style partitioning for DuckDB partition pruning.
    Structure: duckdb_hive/row_chunk=X/hash_bucket=Y/data.parquet
    """
    output_dir = Path(config.output_dir) / "duckdb_hive"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    sensors = generate_sensors(config.num_sensors)
    num_row_chunks = (config.num_rows + config.chunk_rows - 1) // config.chunk_rows

    print(f"Exporting Hive-partitioned for DuckDB: {num_row_chunks} row_chunks x {config.num_sensors} hash_buckets")

    files_written = 0

    for row_chunk in range(num_row_chunks):
        ts_start = row_chunk * config.chunk_rows + 1
        ts_end = min((row_chunk + 1) * config.chunk_rows, config.num_rows)

        for sensor in sensors:
            hash_bucket = get_hash_bucket(sensor, config.num_sensors)

            # Create partition directory
            partition_dir = output_dir / f"row_chunk={row_chunk}" / f"hash_bucket={hash_bucket}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            query = f"""
                SELECT * FROM iot
                WHERE timestamp >= {ts_start}
                  AND timestamp <= {ts_end}
                  AND sensor_id = '{sensor}'
            """
            result = conn.execute(query).fetch_arrow_table()

            if result.num_rows == 0:
                continue

            filepath = partition_dir / "data.parquet"
            pq.write_table(result, filepath, compression='snappy')
            files_written += 1

    return files_written


def save_config(config: DataConfig):
    """Save configuration for Rust benchmark."""
    output_dir = Path(config.output_dir)
    config_path = output_dir / "config.json"
    config.to_json(str(config_path))


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark data')
    parser.add_argument('--rows', type=int, default=1_000_000, help='Number of rows')
    parser.add_argument('--sensors', type=int, default=5, help='Number of sensors (hash buckets)')
    parser.add_argument('--chunk-rows', type=int, default=10_000, help='Rows per chunk')
    parser.add_argument('--chunk-cols', type=int, default=4, help='Number of column groups')
    parser.add_argument('--data-columns', type=int, default=10, help='Number of metric columns')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='./benchmark_data', help='Output directory')
    parser.add_argument('--config', type=str, help='Load config from JSON file')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for DuckDB inserts')

    args = parser.parse_args()

    if args.config:
        config = DataConfig.from_json(args.config)
    else:
        config = DataConfig(
            num_rows=args.rows,
            num_sensors=args.sensors,
            num_data_columns=args.data_columns,
            chunk_rows=args.chunk_rows,
            chunk_cols=args.chunk_cols,
            seed=args.seed,
            output_dir=args.output,
        )

    batch_size = args.batch_size

    print("\n" + "=" * 60)
    print("DATA GENERATION CONFIG")
    print("=" * 60)
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  batch_size: {batch_size:,}")
    print()

    # Clean output directory
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    t0 = time.time()

    # Create DuckDB and populate
    db_path = output_dir / "benchmark.duckdb"
    conn = duckdb.connect(str(db_path))
    create_duckdb_table(conn, config)
    populate_duckdb(conn, config, batch_size)

    conn.execute("CREATE TABLE iot_ordered AS SELECT * FROM iot ORDER BY sensor_id, timestamp")
    conn.execute("DROP TABLE iot")
    conn.execute("ALTER TABLE iot_ordered RENAME TO iot")
    

    insert_time = time.time() - t0
    row_count = conn.execute("SELECT COUNT(*) FROM iot").fetchone()[0]
    print(f"\nDuckDB: {row_count:,} rows in {insert_time:.2f}s")

    # Export chunks for Rust
    t1 = time.time()
    files, chunk_bytes = chunk_and_write(conn, config)
    export_time = time.time() - t1

    # Export Hive-partitioned for DuckDB partition pruning
    t2 = time.time()
    hive_files = export_hive_partitioned(conn, config)
    hive_time = time.time() - t2
    print(f"Hive export: {hive_files} files in {hive_time:.2f}s")

    # Save config
    save_config(config)
    conn.close()

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total rows: {config.num_rows:,}")
    print(f"  DuckDB file: {db_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Chunk files: {files}")
    print(f"  Chunk total size: {chunk_bytes / (1024*1024):.1f} MB")
    print(f"  Insert time: {insert_time:.2f}s")
    print(f"  Export time: {export_time:.2f}s")
    print(f"  Total time: {time.time() - t0:.2f}s")


if __name__ == '__main__':
    main()
