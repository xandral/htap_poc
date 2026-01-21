# NDim: Coordinate-Based HTAP Storage Engine

**An experimental storage engine unifying SQL-style queries and vector operations through multi-dimensional coordinate addressing.**

---

## Abstract

NDim explores an alternative approach to hybrid transactional/analytical processing (HTAP) by replacing traditional index structures with coordinate-based file addressing inspired by array file formats (Zarr, NetCDF). The system partitions data across multiple dimensions (row chunks, hash buckets) and locates files through direct coordinate calculation rather than index traversal.

The engine is going to support merge-on-read updates with serializable expressions, enabling complex derived updates without blocking readers. Vector operations (slicing, reduction, similarity search) execute natively during scans, avoiding the need for separate vector databases or post-processing pipelines.

**Status:** Experimental prototype.

---

## Key Design Aspects

### 1. Multi-Dimensional Coordinate Addressing

Files are located through coordinate calculation across four dimensions:

```
File: chunk_r{row}_c{col}_h{hash}_rg{range}_v{version}.parquet

Dimensions:
- Row:    global_id // chunk_rows        (horizontal partitioning)
- Column: column_index // chunk_cols     (vertical chunking)
- Hash:   hash(value) % buckets          (distribution on categorical columns)
- Range:  value // interval              (time/numeric range partitioning)
```

Given a query with filters on indexed dimensions, the system calculates which coordinate ranges contain matching data and reads only those files. 

**Trade-off:** Coordinate calculation is fast, but without statistics the system may read files that contain no matching rows. The approach works best when filters align with partition boundaries.

### 2. Native Vector Operations

Vector operations execute during chunk scans rather than as post-processing:

```python
result = store.scan(
    filters=[("status", "=", "active")],
    vector_ops=[
        lambda t: VectorOps.cosine_similarity_query(t, "embedding", query_vec, "score")
    ],
    columns=["user_id", "score"]
)
```

Supported operations:
- **Similarity:** cosine similarity (query vs column, pairwise)
- **Slicing:** N-dimensional tensor slicing with NumPy semantics
- **Reduction:** mean, sum, max, min across array elements
- **Filtering:** condition on array elements (e.g., `array[i] > threshold`)
- **Arithmetic:** element-wise operations, scalar multiplication

For fixed-length 1D vectors, the system reshapes data into matrices for vectorized NumPy operations. Variable-length and N-D arrays fall back to per-row evaluation.

### 3. Merge-on-Read with Serializable Expressions

Updates write patch files rather than modifying base data:

```python
store.update(
    filters=[("region", "=", "EU")],
    updates={
        "price": col("price") * 1.15,           # Expression: 15% increase
        "status": col("new_status"),            # Column reference
        "updated_at": 1704067200                # Scalar value
    }
)
```

The expression tree serializes to JSON:
```json
{
  "type": "arithmetic",
  "left": {"type": "column_ref", "name": "price"},
  "op": "*",
  "right": {"type": "scalar", "value": 1.15}
}
```

During subsequent reads, patches apply in transaction-ID order. Each file version is tagged with a transaction ID, ensuring readers see a consistent snapshot based on their read timestamp. The merged result caches in memory for repeated access.

**Properties:**
- Writers never block readers (append-only patches)
- Expressions evaluate at read time with current column values
- Patches survive process restarts (JSON persistence)
- Version-based chunk selection ensures transactional consistency

**Caching:**
After a chunk is read and patches are applied, the merged result is cached in memory. Currently, cache growth is unbounded (no eviction policy). Cache management is ongoing work.

**Ongoing development:**
- Compaction (consolidate patches into base files)
- Write-ahead log for crash recovery

### 4. Vertical Chunking with Column Groups

Unlike pure columnar formats (1 file per column), NDim groups ~32 columns per file:

```
columns 0-31   -> chunk_r*_c0_*.parquet
columns 32-63  -> chunk_r*_c1_*.parquet
...
```

**Rationale:** HTAP workloads often read multiple related columns together. Grouping reduces file open overhead for multi-column point lookups while maintaining reasonable selectivity for analytical scans.

The system tracks which columns reside in which chunk index, enabling projection pushdown to skip irrelevant column groups.

---

## Architecture

### Component Overview

```
+---------------------------------------+
|  DDIMSession / DDIMFrame (API)        |
|  - Query builder pattern              |
|  - Filter/projection management       |
|  - Lazy evaluation until collect()    |
+------------------+--------------------+
                   |
+------------------v--------------------+
|  NDimStorage Engine                   |
|  - Coordinate calculation             |
|  - File pruning via metadata          |
|  - Parallel I/O (ThreadPoolExecutor)  |
|  - Patch application (merge-on-read)  |
+------------------+--------------------+
                   |
     +-------------+-------------+
     |             |             |
+----v----+  +-----v-----+  +----v----+
| Schema  |  | Sequence  |  | Filter  |
| Manager |  | Manager   |  | Engine  |
+---------+  +-----------+  +---------+
  Column      Global ID,      Expression
  metadata    TID allocation  evaluation

                   |
+------------------v--------------------+
|  Parquet Files (immutable chunks)     |
|  + Patch Files (JSON, append-only)    |
+---------------------------------------+
```

### Data Flow

**Write Path:**
1. Allocate global IDs (if no incremental primary key given) and transaction ID
2. Compute partition coordinates for each row
3. Group rows by (row_chunk, col_chunk, hash_bucket, range_bucket)
4. Write/append to parquet files (parallel)
5. Update metadata asynchronously

**Read Path:**
1. Parse filters, identify pruning candidates
2. Calculate coordinate ranges from filter values
3. Load candidate files (parallel), apply filter pushdown
4. Apply pending patches (merge-on-read)
5. Execute vector operations
6. Join vertical chunks via global_id
7. Project final columns, apply limit

**Update Path:**
1. Allocate transaction ID
2. Serialize filter expressions and update values
3. Write patch file (JSON)
4. Cache patch in memory for immediate visibility

---

## Comparison with Existing Systems

| Aspect | Delta Lake | Apache Iceberg | TileDB | NDim |
|--------|-----------|----------------|--------|------|
| Update model | Merge-on-write | Merge-on-write | In-place | Merge-on-read |
| Index structure | File-level stats | Partition pruning | Coordinates | Coordinates |
| Vector operations | External | External | Native | Native |
| Multi-dimensional | No | No | Yes | Yes |
| Column layout | Per-file | Per-file | Per-attribute | Column groups |
| Compaction | Required | Required | Optional | In progress |
| ACID | Yes | Yes | Yes | Partial (WAL in progress) |

**Key differences:**

- **vs Delta/Iceberg:** NDim uses coordinate addressing instead of manifest-based file tracking. Updates are merge-on-read rather than merge-on-write, trading read latency for write speed.

- **vs TileDB:** Similar coordinate-based approach, but NDim uses column groups rather than single-column files, and integrates SQL-style filtering with vector operations in a unified scan.

---

## Design Decisions

### Why Merge-on-Read?

Merge-on-write (Delta, Iceberg) rewrites affected files during updates, maintaining read performance but slowing writes and requiring compaction. Merge-on-read defers this cost to read time.

**Suitable for:**
- Write-heavy workloads with eventual reads
- Analytical queries tolerant of small latency increases
- Systems where write availability matters more than read latency

**Not suitable for:**
- High-frequency point lookups on frequently updated data
- Workloads requiring consistent sub-millisecond reads

### Why Column Groups?

Pure columnar (1 column = 1 file) optimizes analytical scans but penalizes multi-column lookups with many file opens. Row-store formats optimize lookups but waste I/O on wide-table scans.

Column groups (~32 columns per file) could balance both:
- Point lookups: 1-4 files instead of 100+ columns
- Analytical scans: Skip irrelevant column groups

The grouping is currently static (insertion order). Future work could use access pattern analysis for intelligent grouping.

### Why Coordinate Addressing?

Traditional indexes (B-tree, LSM) require maintenance overhead and can become bottlenecks under high write loads. Coordinate addressing computes file locations from filter values directly.

**Advantages:**
- No index maintenance during writes
- Predictable file locations
- Natural support for multi-dimensional queries

**Disadvantages:**
- Requires filters on partitioned columns for effective pruning
- No statistics for selectivity estimation
- May read empty files when data is sparse

---

## Current Status and Limitations

### Ongoing Development

| Feature | Status | Description |
|---------|--------|-------------|
| Compaction | In progress | Background consolidation of patches into base files |
| Write-ahead log (WAL) | In progress | Crash recovery with checkpoint mechanism |
| JOIN support | In progress | Multi-table query capability |
| GROUP BY / aggregation | In progress | Pushdown aggregation to storage layer |
| Cache eviction | In progress | Bounded cache with LRU or similar policy |

### Current Limitations

| Issue | Impact |
|-------|--------|
| No query optimizer | Suboptimal execution plans |
| File size unbounded | Skewed data can cause large files |
| Limited transaction isolation | Read-committed only |
| No schema evolution | Column changes require rewrite |

### Performance Constraints

| Constraint | Cause | Potential Solution |
|------------|-------|-------------------|
| Python GIL | Limits true parallelism | Rust rewrite for compute-heavy paths |
| PyArrow I/O overhead | File open latency | Memory-mapped files, buffer pool |
| Patch accumulation | Linear read slowdown | Compaction (ongoing) |

---

## API Overview

### Session and Dataset Creation

```python
from src.core.api import DDIMSession, col

session = DDIMSession("./data")

# Create dataset with partitioning config
df = session.create_dataset(
    "sensors",
    chunk_rows=100_000,
    chunk_cols=32,
    hash_dims={"sensor_type": 10},
    range_dims={"timestamp": 3600}
)
```

### Write

```python
import pyarrow as pa

data = pa.Table.from_pydict({
    "sensor_id": [1, 2, 3],
    "sensor_type": ["temp", "pressure", "temp"],
    "timestamp": [1000, 1001, 1002],
    "value": [23.5, 101.3, 24.1],
    "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})

df.write(data)
```

### Query with Filters and Vector Operations

```python
result = (
    df.filter(col("sensor_type") == "temp")
      .filter(col("timestamp") >= 1000)
      .with_column("similarity", col("embedding").cosine_sim([0.1, 0.2]))
      .select("sensor_id", "value", "similarity")
      .collect()
)
```

### Update with Expressions

```python
df.filter(col("sensor_type") == "temp").update({
    "value": col("value") * 1.05,  # 5% calibration adjustment
    "calibrated": True
})
```

---

## Project Structure

```
src/
  core/
    api.py              # DDIMSession, DDIMFrame, expression DSL
    storage.py          # NDimStorage engine, partitioning, merge-on-read
    vector_operations.py # VectorOps, tensor operations, serialization
    expressions.py      # ColumnRef, ArithmeticExpr, ExpressionEvaluator
  utils/
    filters.py          # FilterEngine for expression evaluation
    schema_manager.py   # Column metadata, chunk indexing
    sequence_manager.py # ID allocation, file versioning
    coordinates.py      # Hash/range bucket calculation
```

---

## Planned Improvements

### Adaptive Hash Partitioning for Skew Handling

Current hash partitioning assumes uniform distribution. Real-world data often exhibits skew (e.g., 80% of rows in 2 buckets).

**Approach:**
- Maintain per-value counters during writes
- When a bucket exceeds threshold, split into sub-buckets using secondary hash
- Alternative: map categorical values to numeric range based on frequency

```
Before: hash("US") % 5 = 2  -> all US rows in bucket 2 (skewed)
After:  hash("US") % 5 = 2, sub_bucket = row_count("US") // max_bucket_size
        -> US rows distributed across 2.0, 2.1, 2.2, ...
```

### Sparse Data Catalog

Coordinate addressing assumes dense data across dimensions. Sparse data (many empty coordinate combinations) causes unnecessary file existence checks.

**Approach:**
- Maintain lightweight catalog of existing coordinate combinations
- Bitmap or bloom filter per dimension for fast pruning
- Update catalog incrementally during writes

```
Without catalog:
  Query: region='Antarctica' -> check files for all row chunks (most empty)

With catalog:
  catalog.exists(hash_dim='Antarctica') = False -> skip entirely
```

---

## Research Directions

### 1. Learned Partitioning for Vector Data (Vector Store Adaptation)

This direction explores adapting NDim as a native vector store. SQL-style partitioning (hash, range) is suboptimal for high-dimensional vectors because it ignores semantic similarity. The goal is to replace traditional partition dimensions with learned coordinates derived from the vector space itself.

**Motivation:** Current vector databases (Pinecone, Milvus, Qdrant) use specialized indexes (HNSW, IVF). NDim could offer an alternative where vector partitioning integrates directly into the coordinate-based storage model, enabling unified SQL + vector queries without separate systems.

**Proposed approach:**
- Apply dimensionality reduction (PCA, UMAP) to embeddings during ingestion
- Cluster reduced vectors using k-means, hierarchical clustering, or learned quantization
- Use cluster ID as partition coordinate, replacing hash dimension for vector columns

```
Traditional SQL partitioning:  chunk_h{hash(user_id)}_...
Vector-aware partitioning:     chunk_cl{cluster_id}_...

Hybrid query: SELECT * FROM products
              WHERE category = 'electronics'
              AND embedding <-> query_vec < 0.5

Execution:
  1. Hash pruning on category (SQL dimension)
  2. Cluster pruning on embedding (vector dimension)
  3. Exact similarity search within candidate chunks
```

**Trade-offs:**
- Requires periodic re-clustering as data distribution shifts
- Cluster boundaries may split semantically similar vectors
- Could integrate incremental clustering (BIRCH, online k-means) for streaming data

**Potential advantage over dedicated vector DBs:** Unified storage for structured data + embeddings, single query language, no ETL between SQL DB and vector index.

### 2. Hybrid Partitioning Strategy

Combine multiple approaches based on column characteristics:

| Column Type | Partitioning Strategy |
|-------------|----------------------|
| Categorical (low cardinality) | Hash with skew detection |
| Categorical (high cardinality) | Frequency-based numeric mapping |
| Numeric/Temporal | Range partitioning |
| Vector/Embedding | Cluster-based partitioning |

The system could infer optimal strategy from schema metadata and data statistics, adapting as distribution changes.

---

## References and Inspiration

- **Zarr:** Chunked, compressed N-dimensional arrays with coordinate-based addressing
- **Delta Lake / Iceberg:** Table formats with versioning (contrast: merge-on-write)
- **DuckDB:** Embedded analytical database (benchmark comparison target)
- **SQL:2023 MDA:** Multidimensional array extensions to SQL standard
- **LSH / FAISS:** Locality-sensitive hashing for approximate vector search - inspiration for cluster-based partitioning


---

Email: alessandro.ariu95@gmail.com


---

