# NDim HTAP Storage Engine (PoC)

![Status](https://img.shields.io/badge/Status-Experimental%20%2F%20PoC-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Dependencies](https://img.shields.io/badge/Backend-PyArrow%20%7C%20NumPy-yellowgreen)

**Experimental attempt to unify SQL tables and vector operations in a single storage layer.**

---

## Overview

This PoC explores whether coordinate-based indexing (inspired by Zarr) can support both SQL queries and vector operations without separate systems or ETL pipelines.

**Goal:** Query like this in one pass:
```python
# SQL filter + vector operation
store.scan(
    filters=[("status", "=", "active")],
    vector_ops=[lambda t: cosine_similarity(t, "embedding", query_vec)]
)
```

**Status:** Basic implementation works, many features missing. Rudimentary but functional.

---

## Core Concepts

### 1. Coordinate-Based Files (Zarr-Inspired)

File location calculated, not looked up:
```
chunk_r{row}_c{col}_h{hash}_rg{range}_v{version}.parquet
```

**Example:**
```
Query: WHERE user_id = 150000 AND region = 'US-WEST'

Calculation:
- Row:  150000 // 100_000 = 1        → r1
- Hash: hash('US-WEST') % 10 = 7     → h7
- Range: (if timestamp indexed)      → rg*
- Col: (needed columns)              → c*

Files: chunk_r1_c*_h7_rg*_v2.parquet
```

No B-tree index—pure arithmetic.

### 2. Vector Operations (SQL:2023 MDA-Inspired)

Operations execute during chunk scans:

**Slicing:**
- `mda_slice_tensor()` - 1D (PyArrow fast) or N-D (NumPy)

**Reduction:**
- `mda_reduce_mean/sum/max/min()` - Aggregate across dimensions

**Similarity:**
- `cosine_similarity_query()` - Compare vs single vector
- `cosine_similarity()` - Pairwise column comparison

**Arithmetic:**
- `scalar_op()` - Multiply/add/divide by scalar
- `element_wise_add/mul()` - Combine arrays
- `math_n_sum()` - Sum multiple arrays

**Filtering:**
- `mda_filter_by_index()` - Filter rows where `array[idx] meets condition`

All return `pa.Table` for chaining.

### 3. Merge-on-Read Updates (Rudimentary)

**Update Flow:**
1. **Write Patch:** Update creates JSON patch file with transaction ID (tid)
   ```json
   {
     "tid": 42,
     "filters": [("region", "=", "US")],
     "updates": {"status": "inactive"}
   }
   ```

2. **Scan-Time Merge:** 
   - Load base parquet file
   - Apply all patches where `tid <= current_tid`
   - Cache merged result in memory

3. **Compaction (NOT IMPLEMENTED):**
   - Should consolidate patches → new base file
   - Requires WAL for crash safety
   - Currently: patches accumulate forever

**Properties:**
- ✅ Writers don't block readers (append-only patches)
- ✅ Immediate consistency (read-after-write works)
- ❌ No WAL (crash = lost patches)
- ❌ No vacuum (storage grows)
- ⚠️ Query overhead grows with patch count

---

## Quick Example

```python
import pyarrow as pa
from src.core.storage import NDimStorage
from src.core.vector_operations import VectorOps

# Setup
store = NDimStorage(
    "./data",
    chunk_rows=100_000,
    chunk_cols=32,
    hash_dims={"region": 10},
    range_dims={"timestamp": 3600}
)

# Insert
data = pa.Table.from_pydict({
    "user_id": [1, 2, 3],
    "region": ["US", "EU", "US"],
    "status": ["active", "active", "inactive"],
    "embedding": [[0.9, 0.1], [0.1, 0.8], [0.5, 0.5]]
})
store.write_batch(data)

# Query: SQL + Vector
result = store.scan(
    filters=[("status", "=", "active")],
    vector_ops=[
        lambda t: VectorOps.cosine_similarity_query(
            t, "embedding", [1.0, 0.0], "similarity"
        )
    ],
    columns=["user_id", "similarity"]
)

# Update (writes patch log)
store.update(
    filters=[("region", "=", "US")],
    updates={"status": "inactive"}
)
# Next scan auto-applies patches
```

---

## Architecture

```
┌─────────────────────────────────────┐
│  DDIMSession API (SQL-like)         │
│  - filter(), select(), with_column()│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  NDimStorage Engine                 │
│  - Coordinate calculation           │
│  - Patch log (merge-on-read)        │
│  - Vector ops during scan           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Parquet Files (immutable chunks)   │
│  chunk_r{row}_c{col}_h{hash}...     │
└─────────────────────────────────────┘
```

---

## Known Limitations

**Critical Missing:**
- ❌ No compaction (patches accumulate)
- ❌ No WAL (crash unsafe)
- ❌ File size can explode with skewed data
- ❌ No query optimizer

**Performance:**
- ⚠️ Python GIL limits parallelism
- ⚠️ Coordinate calculation O(1) in theory, but file I/O overhead unclear
- ⚠️ Not benchmarked vs real systems

**Vector Ops:**
- ⚠️ Basic operations only (no matmul, complex reshaping)
- ⚠️ No query planning (naive per-chunk execution)

---

## Design Choices

### Column Groups vs Pure Columnar

**TileDB/SciDB:** 1 column = 1 file  
**This PoC:** ~32 columns per file

**Why:** Multi-column point lookups need fewer file opens 

**Target:** Hybrid workloads (business apps, not pure analytics)

### Merge-on-Read vs Merge-on-Write

**Delta Lake/Iceberg:** Compact during writes  
**This PoC:** Merge during scans

**Why:** Fast updates (append patches), acceptable for analytical queries. Simpler to implement.


---

## Installation

```bash
pip install -r requirements.txt
```

---

## Roadmap

**Phase 1 (Critical):**
- [ ] Compaction with transaction log
- [ ] Write-Ahead Log (WAL)
- [ ] File size bounds
- [ ] Link type storage for external tensors

**Phase 2:**
- [ ] Query optimizer
- [ ] Parallel execution
- [ ] Better caching

**Phase 3 (Long Term):**
- [ ] Rust rewrite (no GIL)
- [ ] Distributed execution

---

## Contact

mail: alessandro.ariu95@gmail.com