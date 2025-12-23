# NDim HTAP Storage Engine (PoC)

![Status](https://img.shields.io/badge/Status-Experimental%20%2F%20PoC-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Dependencies](https://img.shields.io/badge/Backend-PyArrow%20%7C%20NumPy-yellowgreen)

**A Multi-Dimensional Columnar Storage Engine inspired by Zarr, with native SQL:2023 MDA vector support.**

---

### ⚠️ Disclaimer: Proof of Concept

> **PLEASE NOTE: This project is currently under active development. It is a strictly experimental Proof of Concept (PoC) designed to demonstrate architectural patterns.** >
> **Features are incomplete, bugs are present. Use only for research and testing purposes.**

---

## Overview

NDim HTAP explores a hybrid approach to data storage, merging the analytical power of columnar databases (OLAP) with the N-dimensional flexibility of scientific array computing.

The architecture abandons centralized B-Tree indexes in favor of a **coordinate-based access pattern** inspired by the **Zarr** format. Every row and column is mathematically mapped to a physical file, enabling parallel reads and immediate data pruning.

### Unified HTAP Architecture
**Goal:** Achieve an **Hybrid Transactional/Analytical Processing (HTAP)** system.
Unlike traditional systems that separate operational data (SQL) from AI data (Vectors), this engine unifies them:

* **Transactional (OLTP):** Supports fast lookups, updates by ID, and consistency using Hash Indexing.
* **Analytical (OLAP):** Supports massive scans, tensor slicing, and vector math natively.
* **The Benefit:** Filter by SQL metadata (e.g., `status='active'`) and compute complex Vector math (e.g., `cosine_similarity`) in a **single, zero-copy pass**.

## Core Architecture

The system organizes data into sparse "Hypercubes" stored as Parquet files. The physical location of any data point is deterministically calculated via its coordinates, removing the need for centralized lookup indexes.

### 1. The Coordinate System (Zarr-inspired)
Instead of scanning indexes, the engine calculates the file path using a grid system:
`chunk_r{row}_c{col}_h{hash}_rg{range}_v{ver}.parquet`

| Dimension | Role | Description |
| :--- | :--- | :--- |
| **Row Index** | Primary Key | Sequential grouping (e.g., 100k rows per chunk). |
| **Column Group** | Optimization | Vertical partitioning to reduce I/O on wide tables. |
| **Hash Dim** | Sharding | Distribution based on key hash (e.g., `user_id`). |
| **Range Dim** | Pruning | Ordered partitioning (e.g., `timestamp`) for time-series. |

###  Core Indexing Logic:

The system calculates the physical coordinate (File Path + Row Offset) using the configured chunk sizes.

#### The Mapping Formula
For a given `Primary_Key` (e.g., a timestamp or auto-increment ID) and a `Column_Index`:

**1. Vertical Mapping (Which Row?)**
Determine the file partition and the specific row inside it:
* **File Partition** = `Primary_Key // Chunk_Rows`
* **Internal Row Offset** = `Primary_Key % Chunk_Rows`

**2. Horizontal Mapping (Which Column Group?)**
Determine which column chunk file to open:
* **Column Group** = `Column_Index // Chunk_Cols`
* **Internal Column Offset** = `Column_Index % Chunk_Cols`

---

#### Concrete Example
Let's assume the system is configured with:
* `CHUNK_ROWS = 1000` (rows per file)
* `CHUNK_COLS = 10` (columns per file)

If we want to retrieve **User ID 2505** and **Column 23**:

1.  **Partition Calculation:** `2505 // 1000 = 2` → The system opens the file for **Partition 2**.
2.  **Row Seek:** `2505 % 1000 = 505` → The system seeks directly to **Row 505**.
3.  **Column Group:** `23 // 10 = 2` → The system reads from **Col-Group 2**.

**Result:** The engine opens `partition_r2_c2.parquet` and reads row `505` instantly.

### 2. SQL:2023 MDA Exploration
This project includes an **experimental implementation** of the Multi-Dimensional Arrays (MDA) concepts introduced in the SQL:2023 standard. 

Rather than treating arrays strictly as unrelated lists (like `ARRAY[]` in older SQL), this engine attempts to treat columns as **mathematical tensors**.

## Concurrency: MVCC & Merge-on-Read

NDim HTAP handles concurrency using a **Multi-Version Concurrency Control (MVCC)** model combined with a **Merge-on-Read** strategy. This ensures that readers are never blocked by writers (`Readers don't block Writers`).

### 1. Immutability & Versioning
Data chunks on disk are **immutable**. When an update occurs (e.g., updating a user's embedding or status), the engine does not modify the existing Parquet file. Instead:

1.  **Read:** The affected chunk is loaded into memory.
2.  **Patch/Apply:** The update is applied to the in-memory PyArrow table.
3.  **Write New Version:** A **new file** is written with an incremented version number.
    * *Old:* `chunk_r1_c0_h5_..._v1.parquet`
    * *New:* `chunk_r1_c0_h5_..._v2.parquet`

### 2. The Catalog (Merge-on-Read Logic)
The `SequenceManager` maintains a lightweight **File Catalog** (JSON-based in this PoC) acting as the source of truth.

* **Snapshot Isolation:** When a `DDIMSession` starts a read query, it takes a "snapshot" of the active file versions.
* **Logical Merge:** Even if a writer creates `v3` while a reader is processing `v2`, the reader continues to use `v2` referenced in its snapshot. The "Merge" happens logically by resolving the latest committed version ID for each coordinate before scanning.

### 3. Vacuuming (Garbage Collection)
Since updates generate new files, the storage grows over time. A `vacuum()` process (planned) scans the catalog to identify and physically delete "stale" versions (e.g., `v1` when `v2` is active and no active queries rely on `v1`).

## Quick Start Example

The `DDIMSession` API provides a lazy-execution interface similar to Spark.

```python
fimport pyarrow as pa
from src.core.storage import NDimStorage
from src.core.vector_operations import VectorOps

# 1. SETUP HTAP STORAGE
# We configure 'hash_dims' for fast SQL-like lookups (OLTP)
# The engine automatically handles the Vector data (OLAP)
store = NDimStorage("./demo_htap", hash_dims={"user_id": 4})

# 2. INSERT HYBRID DATA
# We mix standard SQL columns with AI Tensor columns
data = {
    "user_id": [1, 2, 3],              # SQL: Primary Key
    "role": ["admin", "user", "user"], # SQL: Metadata
    "face_embedding": [                # TENSOR: AI Data (Vectors)
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ]
}
store.write_batch(pa.Table.from_pydict(data))

# 3. RUN HYBRID QUERY
# Scenario: Find 'users' (SQL) and calculate their vector magnitude (Tensor)
print("--- HTAP Query Results ---")

def compute_magnitude(table):
    # Perform math on the tensor column on-the-fly
    return VectorOps.mda_reduce_sum(table, "face_embedding")

result = store.scan(
    # SQL FILTER: Standard WHERE clause
    filters=[("role", "=", "user")], 
    
    # TENSOR COMPUTE: Math operation
    vector_ops=[compute_magnitude],
    
    columns=["user_id", "face_embedding"]
)

# Output shows SQL ID + Computed Tensor Result
print(result.to_pandas())
#    user_id  face_embedding_sum
# 0        2                 1.0
# 1        3                 1.0
```


## Limitations & Known Issues

As a PoC, the engine has strict boundaries. Many features are architectural skeletons awaiting full implementation.

* **Missing Compaction (Vacuum):**
    The system currently uses a "Merge-on-Read" strategy, creating new file versions (`_v2`, `_v3`) for every update.
    * *Current Issue:* The `vacuum()` logic to physically merge old files and reclaim space is **not yet implemented**. The storage footprint grows indefinitely with updates.


* **Incomplete Vector Operations:**
    While the interface supports SQL:2023 MDA syntax, the backend implementation is limited.
.

* **Naive Update Mechanism:**
    Updates follow a "Happy Path". There is no rollback mechanism if a write fails halfway through.
    * *Concurrency:* While file-locking exists, it is not robust against process crashes (No Write-Ahead Log).
    * *Error Handling:* Type mismatches or I/O errors may leave the dataset in an inconsistent state.

* **Dependencies:**
    Heavy reliance on the Python Global Interpreter Lock (GIL) limits true parallelism during query execution.

##  Roadmap: Engineering & Performance

The goal is to evolve from a prototype to a stable, performant engine.

### Phase 1: Code Quality & Stability
- [ ] **Code Hardening:** Refactor the codebase to handle edge cases, type mismatches, and file corruption gracefully.
- [ ] **Structured Logging:** Replace `print` debugging with a proper logging rotation system.
- [ ] **Testing Suite:** Implement a comprehensive Unit and Integration Test suite.

### Phase 2: Core Performance Optimization
- [ ] **Smart Caching:** Implement a Cache for Chunk Metadata and frequent file handles to reduce filesystem metadata operations (`stat`/`open`).
- [ ] **I/O Optimization:** 
- [ ] **Memory Management:** Refactor the pipeline to enforce strictly **Zero-Copy** semantics where possible, passing pointers between PyArrow and NumPy without duplication.

### Phase 3: The Rust Port (Long Term)
- [ ] **Rewrite Core in Rust:** To solve the GIL and performance issues definitively, the storage engine core (IO, Locking, Coordinates) will be ported to Rust, exposing a high-level Python binding.
