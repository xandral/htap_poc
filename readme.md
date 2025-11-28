# NDim HTAP Storage Engine - POC

**A Multi-Dimensional, Columnar Storage Engine for HTAP Workloads**  
*Python Proof-of-Concept → Future Rust Implementation*

---

## Table of Contents

1. [Genesis & Inspiration](#1-genesis--inspiration)
2. [Core Concept: From Zarr to Tabular](#2-core-concept-from-zarr-to-tabular)
3. [Architecture Overview](#3-architecture-overview)
4. [N-Dimensional Partitioning Strategy](#4-n-dimensional-partitioning-strategy)
5. [MVCC Implementation](#5-mvcc-implementation)
6. [Performance Model](#6-performance-model)
7. [Quick Start](#7-quick-start)


---

## 1. Genesis & Inspiration

### The Zarr Foundation

**Zarr** is a format for storing chunked, compressed N-dimensional arrays, widely used in scientific computing (climate models, genomics, satellite imagery).

**Core principle**: Split large arrays into small rectangular chunks indexed by coordinates.

```
3D Array: Temperature(Latitude, Longitude, Time)
Shape: [1000, 2000, 365]

Chunked as: [100, 100, 30] blocks
Result: 10 × 20 × 13 = 2,600 files

File naming: /data/0.0.0, /data/0.1.0, ...
           coordinate: (lat_chunk, lon_chunk, time_chunk)

Query: "Temperature for Rome (lat=41, lon=12) in January"
→ Calculate chunk: (4, 0, 0)
→ Read 1 file instead of 2,600
```

**Key insight**: Coordinate-based addressing enables O(1) file lookup, minimal I/O.

---

### Mapping to Tabular Data

**Question**: Can we apply this to relational tables?

**Conceptual mapping**:

| Zarr (Arrays) | NDim HTAP (Tables) |
|---------------|--------------------|
| Latitude dimension | Row Chunk ID |
| Longitude dimension | Column Chunk ID |
| Time dimension | Transaction Version (MVCC) |
| Extra dimensions | Hash/Range partitioning |
| Chunk coordinate | File path |

**Result**: Tables chunked into small rectangles addressable by multi-dimensional coordinates.

---

## 2. Core Concept: From Zarr to Tabular

### Problem Statement

Traditional databases have opposing trade-offs:

**OLTP (Row-oriented)**:
- Storage: `[Row1: col1,col2,...,colN | Row2: ... | Row3: ...]`
- Fast: Point lookups (`WHERE id=X`)
- Slow: Analytics (`SELECT AVG(col)` requires full scan)

**OLAP (Column-oriented)**:
- Storage: `[Col1: all_values | Col2: all_values | ...]`
- Fast: Aggregations (`SELECT AVG(col)`)
- Slow: Row reconstruction

**HTAP Goal**: Support both workloads without ETL pipeline.

---

### NDim HTAP Approach

**Insight**: Chunk tables in multiple dimensions simultaneously.

```
Original Table: 1M rows × 100 columns

Dimension 1: ROW CHUNKS (Horizontal Partitioning)
  Chunk 0: Rows 0-99,999
  Chunk 1: Rows 100,000-199,999
  ...

Dimension 2: COLUMN CHUNKS (Vertical Partitioning)
  Group 0: Columns 0-31
  Group 1: Columns 32-63
  ...

Dimension 3+: USER-DEFINED DIMENSIONS
  Hash: hash(city) % buckets
  Range: timestamp // interval

Result: Each file is a small rectangle
  chunk_r0_c0_h3_r100_v1.parquet
    - Rows: 0-99,999
    - Columns: 0-31
    - Hash bucket: 3
    - Range bucket: 100
    - Version: 1
    - Size: ~10MB (vs 10GB full table)
```

**Query optimization**: Read only files intersecting query predicates.

---

## 3. Architecture Overview

### High-Level Components

```
CLIENT
  |
  v
QUERY ENGINE
  - FilterEngine: Evaluates logic expressions (AND/OR/comparison)
  - PruningEngine: Maps filters to file coordinates
  - ParallelIO: ThreadPool for concurrent reads
  |
  v
CATALOG
  - FileCatalog: Tracks file versions (logical_key -> max_version)
  - SequenceManager: Persistent global_id counter
  - VersionTracker: MVCC coordination
  |
  v
STORAGE LAYER
  - Parquet files (columnar, compressed)
  - Naming: chunk_r{row}_c{col}_h{hash}_r{range}_v{version}.parquet
```

---

## 4. N-Dimensional Partitioning Strategy

### Dimension Types

**PRIMARY DIMENSION: Row Chunk**
- Always present
- Sequential: `global_id // chunk_rows`
- Controls horizontal scaling
- Example: 1M rows with chunk_rows=100k → 10 chunks

**COLUMN DIMENSION: Vertical Partitioning**
- Groups related columns
- Fixed size: chunk_cols (e.g., 32)
- Purpose: Read only needed columns
- Example: 100 columns → 4 column groups

**HASH DIMENSIONS: Categorical Distribution**
- User-defined columns for even distribution
- Calculation: `hash(value) % buckets`
- Best for: High-cardinality categorical (user_id, session_id, country)
- Example: `city` with 10 buckets → hash('Roma') % 10 = 3

**RANGE DIMENSIONS: Ordered Distribution**
- User-defined columns for range queries
- Calculation: `value // interval`
- Best for: Numerical/temporal (timestamp, price, age)
- Example: `timestamp` with interval=86400 → bucket = day_number

**VERSION DIMENSION: MVCC**
- Copy-on-Write versioning
- Monotonically increasing per logical chunk
- Purpose: Snapshot isolation
- Example: Update creates v2, readers still see v1

---

## 5. MVCC Implementation

### Versioning Strategy

**Concept**: Each logical chunk can have multiple versions coexisting on disk.

```
Timeline:
T=0: Write initial data
     chunk_r0_c0_h3_r5_v1.parquet

T=1: Update 10% of rows
     chunk_r0_c0_h3_r5_v2.parquet (new)
     v1 still exists

T=2: Vacuum (garbage collection)
     v1 deleted, v2 remains
```

**Version tracking**:
```python
catalog.active_versions = {
    'chunk_r0_c0_h3_r5': 2,  # Latest version
    'chunk_r0_c1_h3_r5': 1,
    ...
}
```

**Reader isolation**:
- Readers call `get_active_files()` at query start
- Snapshot: List of files with specific versions
- Subsequent writes don't affect this snapshot
- Result: Repeatable reads (similar to PostgreSQL REPEATABLE READ)

---

### Vacuum Process

```python
def vacuum(self):
    with catalog.lock:
        active_map = catalog.active_versions.copy()
    
    for file in directory:
        logical_key, version = parse_filename(file)
        
        if version < active_map[logical_key]:
            file.unlink()  # Delete old version
```

**Properties**:
- Frees disk space
- Safe: Only deletes versions strictly older than active
- Issue: Running readers might still have handles to deleted files
  - Linux: File remains accessible until close (unlink delayed)
  - Windows: Deletion might fail if file is open

---

## 6. Performance Model

### Strengths

**Point Lookups (OLTP)**:
```
Query: SELECT * FROM table WHERE global_id = 42

Coordinate calculation: O(1)
  row_chunk = 42 // 100_000 = 0
  
Files to read: 4 (one per column group, all in chunk_r0)
Total I/O: 40MB (4 files × 10MB)
vs Traditional: 10GB (full table scan)

Speedup: 250x
```

**Range Queries**:
```
Query: SELECT * FROM table WHERE timestamp BETWEEN t1 AND t2

Range bucket calculation: O(1)
  start_bucket = t1 // 86400
  end_bucket = t2 // 86400
  buckets_needed = end_bucket - start_bucket + 1

Files to read: buckets_needed × hash_buckets × col_chunks
vs Traditional: All files

Example: 7-day range, 10 hash buckets, 4 col chunks
  NDim HTAP: 7 × 10 × 4 = 280 files
  Traditional: 10,000 files
  Speedup: 35x
```

**Analytical Scans**:
```
Query: SELECT AVG(age) FROM table

Column pruning:
  Needed: global_id, age
  Files: Only c0 (contains age)
  
Total I/O: 2,500 files (c0 only)
vs Row-oriented: 10,000 files (all columns)

Speedup: 4x (with 4 column groups)
```

---

### Limitations

**Wide Filters** (No Pruning):
```
Query: SELECT * FROM table WHERE status = 'active'

If 'status' not in hash_dims or range_dims:
  No pruning possible
  Must scan all files
  
Performance: Same as traditional (full scan)
```

**Cross-Chunk Column Access**:
```
Query: SELECT col1, col50 FROM table WHERE col1 = X

If col1 in c0 and col50 in c2:
  Must read both column groups
  Cannot skip c1
  
Overhead: Read intermediate column group
```

**Update Performance**:
```
Update small percentage of large file:
  - Read: 10MB
  - Filter: 0.1% matches
  - Write: 10MB (entire file, Copy-on-Write)
  
Overhead: 10MB written for 10KB changed
Amplification factor: 1000x

Mitigation: Smaller chunk_rows
```

** CoW + vacuum**:
Creating a new chunk for every update can be heavy. A patching approach + compaction could be better (actually WIP)

## 7. Quick Start

```python
from ndim_storage import NDimStorage
import pyarrow as pa

# Initialize storage
storage = NDimStorage(
    base_path="./data",
    chunk_rows=100_000,         # 100k rows per chunk
    chunk_cols=32,              # 32 columns per group
    hash_dims={'country': 8},   # 8 hash buckets for country
    range_dims={'timestamp': 86400}  # 1-day range buckets
)

# Write data
table = pa.table({
    'id': [1, 2, 3],
    'country': ['US', 'IT', 'FR'],
    'timestamp': [1234567890, 1234567900, 1234567910],
    'value': [100, 200, 300]
})
storage.write_batch(table)

# Query
result = storage.scan(
    filters=[('country', '=', 'IT'), 'AND', ('timestamp', '>', 1234567890)],
    columns=['id', 'value']
)
print(result.to_pandas())

# Update
storage.update(
    filters=[('country', '=', 'US')],
    updates={'value': 999}
)

# Cleanup old versions
storage.vacuum()

# Shutdown
storage.close()
```

---
## Summary

**NDim HTAP** applies Zarr's multi-dimensional chunking to tabular data, enabling:
- Direct coordinate-based file access (O(1) lookup)
- Hybrid OLTP/OLAP performance
- Copy-on-Write MVCC for concurrency
- Embarrassingly parallel I/O

**Python POC** demonstrates viability with 10-100x speedups on targeted queries.

**Rust port** will unlock true potential with zero-copy operations, SIMD, and async I/O.

**Target use cases**: Analytics dashboards, time-series data, event logs, any workload with multi-dimensional access patterns.
