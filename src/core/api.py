import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from dataclasses import dataclass
from src.core.expressions import ColumnRef, ArithmeticExpr

from src.core.storage import NDimStorage
from src.core.vector_operations import VectorOps
from src.utils.filters import FilterEngine  # Uses the uploaded FilterEngine


# ==========================================
# 1. EXPRESSION ENGINE
# ==========================================


class Expression:
    """Base class for building lazy expressions (filters, columns)."""

    def __init__(self):
        pass

    # Map Python operators to BinaryFilter objects
    def __eq__(self, other):
        return BinaryFilter(self, "=", other)

    def __ne__(self, other):
        return BinaryFilter(self, "!=", other)

    def __gt__(self, other):
        return BinaryFilter(self, ">", other)

    def __lt__(self, other):
        return BinaryFilter(self, "<", other)

    def __ge__(self, other):
        return BinaryFilter(self, ">=", other)

    def __le__(self, other):
        return BinaryFilter(self, "<=", other)

    # Bitwise operators for Logical AND/OR
    def __and__(self, other):
        return BinaryFilter(self, "AND", other)

    def __or__(self, other):
        return BinaryFilter(self, "OR", other)


@dataclass
class BinaryFilter(Expression):
    """
    Intermediate representation of a filter in the API.
    It will be converted to the FilterEngine format during execution.
    """

    left: Any
    op: str
    right: Any


class Column(Expression):
    """Represents a dataset column reference."""

    def __init__(self, name: str):
        self.name = name

        # Operatori aritmetici per update colonnari
    def __add__(self, other):
        return ArithmeticExpr(ColumnRef(self.name), "+", self._wrap(other))
    
    def __sub__(self, other):
        return ArithmeticExpr(ColumnRef(self.name), "-", self._wrap(other))
    
    def __mul__(self, other):
        return ArithmeticExpr(ColumnRef(self.name), "*", self._wrap(other))
    
    def __truediv__(self, other):
        return ArithmeticExpr(ColumnRef(self.name), "/", self._wrap(other))
    
    def __radd__(self, other):
        return ArithmeticExpr(self._wrap(other), "+", ColumnRef(self.name))
    
    def __rsub__(self, other):
        return ArithmeticExpr(self._wrap(other), "-", ColumnRef(self.name))
    
    def __rmul__(self, other):
        return ArithmeticExpr(self._wrap(other), "*", ColumnRef(self.name))
    
    def __rtruediv__(self, other):
        return ArithmeticExpr(self._wrap(other), "/", ColumnRef(self.name))
    
    def _wrap(self, other):
        """Converte Column in ColumnRef per serializzazione."""
        if isinstance(other, Column):
            return ColumnRef(other.name)
        elif isinstance(other, ArithmeticExpr):
            return other
        return other

    def v_slice(self, start: int, stop: int) -> "VectorTransformation":
        """Slices a vector/list column (Lazy)."""
        return VectorTransformation(
            "mda_slice", self.name, (start, stop), f"{self.name}_sliced"
        )

    def v_reduce(self, method: str = "mean") -> "VectorTransformation":
        """Reduces a vector to a scalar (Lazy)."""
        return VectorTransformation(
            f"mda_reduce_{method}", self.name, (), f"{self.name}_{method}"
        )

    def cosine_sim(self, vector_query: List[float]) -> "VectorTransformation":
        """Calculates cosine similarity against a query vector (Lazy)."""
        return VectorTransformation(
            "cosine_similarity_literal", self.name, (vector_query,), "sim_score"
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.v_slice(item.start or 0, item.stop)
        raise ValueError("Column getitem supports slices only")


@dataclass
class VectorTransformation:
    """
    Represents a vector operation in the pipeline.
    Supports chaining via the 'parent' attribute.
    """

    op_name: str
    target_col: str
    args: Tuple
    output_alias: str
    parent: Optional["VectorTransformation"] = None

    def alias(self, name: str):
        self.output_alias = name
        return self

    # --- Chaining Methods ---
    def v_slice(self, start: int, stop: int):
        return VectorTransformation(
            "mda_slice",
            self.output_alias,
            (start, stop),
            f"{self.output_alias}_sliced",
            self,
        )

    def v_reduce(self, method: str = "mean"):
        return VectorTransformation(
            f"mda_reduce_{method}",
            self.output_alias,
            (),
            f"{self.output_alias}_{method}",
            self,
        )

    def cosine_sim(self, vector_query: List[float]):
        return VectorTransformation(
            "cosine_similarity_literal",
            self.output_alias,
            (vector_query,),
            "sim_score",
            self,
        )


def col(name: str) -> Column:
    """Syntactic sugar to select a column."""
    return Column(name)


# ==========================================
# 2. DDIM FRAME (With FilterEngine Integration)
# ==========================================


class DDIMFrame:
    def __init__(self, storage: NDimStorage, table_name: str):
        self._storage = storage
        self._table_name = table_name
        self._axes = list(storage.hash_keys) + list(storage.range_keys)
        if storage.global_id:
            self._axes.append(storage.global_id)

        self._filters: List[BinaryFilter] = []
        self._projection: List[str] = []
        self._transformations: List[VectorTransformation] = []
        self._limit: Optional[int] = None

    def _clone(self):
        """Creates a copy of the frame (Immutability)."""
        new = DDIMFrame(self._storage, self._table_name)
        new._filters = self._filters.copy()
        new._projection = self._projection.copy()
        new._transformations = self._transformations.copy()
        new._limit = self._limit
        new._axes = self._axes
        return new

    def _register_chain(self, t: VectorTransformation):
        """Unrolls the transformation chain and registers steps in order."""
        chain = []
        curr = t
        while curr:
            chain.append(curr)
            curr = curr.parent
        for x in reversed(chain):
            if x not in self._transformations:
                self._transformations.append(x)

    # --- HELPERS ---

    def _to_engine_expr(self, expr: Any) -> Any:
        """
        Recursively converts API BinaryFilter objects into the
        tuple/list format required by FilterEngine.

        Example:
            BinaryFilter(col('a'), '>', 5)  -> ('a', '>', 5)
            BinaryFilter(..., 'AND', ...)   -> [expr1, 'AND', expr2]
        """
        if isinstance(expr, BinaryFilter):
            # Case 1: Logical Operator (AND, OR)
            if expr.op in ["AND", "OR"]:
                return [
                    self._to_engine_expr(expr.left),
                    expr.op,
                    self._to_engine_expr(expr.right),
                ]

            # Case 2: Comparison (Column vs Value)
            if isinstance(expr.left, Column):
                return (expr.left.name, expr.op, expr.right)

        return None

    # --- API ---

    def __getitem__(self, key):
        """Numpy-style slicing: df[42, 10:20]."""
        if isinstance(key, (Column, VectorTransformation)):
            return self.select(key)
        new = self._clone()
        if not isinstance(key, tuple):
            key = (key,)

        for i, val in enumerate(key):
            if i >= len(self._axes):
                break
            dim = self._axes[i]
            if isinstance(val, slice):
                if val.start is not None:
                    new._filters.append(col(dim) >= val.start)
                if val.stop is not None:
                    new._filters.append(col(dim) < val.stop)
            elif val is not Ellipsis:
                new._filters.append(col(dim) == val)
        return new

    def filter(self, expr: BinaryFilter) -> "DDIMFrame":
        new = self._clone()
        new._filters.append(expr)
        return new

    def select(self, *cols) -> "DDIMFrame":
        new = self._clone()
        curr = []
        for item in cols:
            if isinstance(item, str):
                curr.append(item)
            elif isinstance(item, Column):
                curr.append(item.name)
            elif isinstance(item, VectorTransformation):
                new._register_chain(item)
                curr.append(item.output_alias)
        new._projection = curr
        return new

    def with_column(
        self, name: str, expr: Union[Column, VectorTransformation]
    ) -> "DDIMFrame":
        new = self._clone()
        if isinstance(expr, VectorTransformation):
            expr.output_alias = name
            new._register_chain(expr)
            if name not in new._projection:
                new._projection.append(name)
        return new

    def limit(self, n: int) -> "DDIMFrame":
        new = self._clone()
        new._limit = n
        return new

    def write(self, data: Union[pa.Table, Dict[str, list]]):
        """Public API for writing data."""
        self._storage.write_batch(data)

    def delete(self) -> int:
        """
        Delete rows matching the current filters.
        Returns the number of affected chunks.
        """
        # Convert API filters to storage format
        storage_filters = []
        for f in self._filters:
            if isinstance(f, BinaryFilter) and isinstance(f.left, Column):
                storage_filters.append((f.left.name, f.op, f.right))
        
        # Compile vector operations if any
        vector_ops = self._compile_ops() if self._transformations else None
        
        # Execute delete operation
        self._storage.delete(filters=storage_filters, vector_ops=vector_ops)

    def explain(self):
        """Prints the query plan."""
        print(f"=== Plan for '{self._table_name}' ===")
        print(f"Raw Filters: {self._filters}")
        print(f"Vector Pipeline: {[t.op_name for t in self._transformations]}")
        print(f"Projection: {self._projection}")

    # --- EXECUTION ---

    def _compile_ops(self) -> List[Callable]:
        """Compiles declarative transformations into executable vector functions."""
        ops = []
        for t in self._transformations:
            # MDA SLICE
            if t.op_name == "mda_slice":

                def mk_slice(c, s, e, o):
                    return lambda tbl: VectorOps.mda_slice_tensor(
                        tbl, col=c, slices=[slice(s, e)], result_col=o
                    )

                ops.append(mk_slice(t.target_col, t.args[0], t.args[1], t.output_alias))

            # MDA REDUCE
            elif "reduce" in t.op_name:

                def mk_red(c, o):
                    return lambda tbl: VectorOps.mda_reduce_mean(
                        tbl, col=c, result_col=o
                    )

                ops.append(mk_red(t.target_col, t.output_alias))

            # COSINE SIMILARITY (Delegate to VectorOps)
            elif "cosine" in t.op_name:

                def mk_sim(c, q, o):
                    return lambda tbl: VectorOps.cosine_similarity_query(
                        table=tbl, col=c, query_vector=q, result_col=o
                    )

                ops.append(mk_sim(t.target_col, t.args[0], t.output_alias))
        return ops

    def collect(self) -> pa.Table:
        # 1. Identify generated columns (outputs of vector ops)
        generated_aliases = {t.output_alias for t in self._transformations}

        # 2. Separate Filters: Storage (Pushdown) vs Memory (Computed)
        storage_filters = []
        memory_filters = []

        for f in self._filters:
            if isinstance(f, BinaryFilter) and isinstance(f.left, Column):
                if f.left.name in generated_aliases:
                    memory_filters.append(f)
                else:
                    storage_filters.append((f.left.name, f.op, f.right))
            else:
                # Complex logic goes to memory filter
                memory_filters.append(f)

        # 3. Columns to fetch
        cols_to_fetch = set()
        for t in self._transformations:
            is_intermediate = False
            for parent in self._transformations:
                if parent.output_alias == t.target_col:
                    is_intermediate = True
            if not is_intermediate:
                cols_to_fetch.add(t.target_col)

        if self._projection:
            for p in self._projection:
                if p not in generated_aliases:
                    cols_to_fetch.add(p)

        cols_list = list(cols_to_fetch) if cols_to_fetch else None

        # 4. EXECUTE SCAN (Pushdown Filters + Vector Ops)
        result = self._storage.scan(
            filters=storage_filters, columns=cols_list, vector_ops=self._compile_ops()
        )

        # 5. MEMORY FILTERS (Using FilterEngine)
        if memory_filters:
            engine_exprs = []
            for f in memory_filters:
                conv = self._to_engine_expr(f)
                if conv:
                    engine_exprs.append(conv)

            if engine_exprs:
                # FilterEngine.evaluate returns a Boolean Mask
                mask = FilterEngine.evaluate(result, engine_exprs)
                if mask is not None:
                    result = result.filter(mask)

        # 6. Projection
        if self._projection:
            final_cols = [c for c in self._projection if c in result.column_names]
            if final_cols:
                result = result.select(final_cols)

        # 7. Limit
        if self._limit:
            result = result.slice(0, self._limit)

        return result

    def show(self, n=5):
        print(self.limit(n).collect().to_pandas())

    def to_pandas(self):
        return self.collect().to_pandas()
    
    def update(self, updates: dict) -> "DDIMFrame":
        """
        Update rows matching current filters.
        
        Args:
            updates: Dict mapping column names to new values.
                     Values can be:
                     - Scalars: {"status": "inactive"}
                     - Column refs: {"col_a": col("col_b")}
                     - Expressions: {"col_a": col("col_b") * 2 + 10}
        
        Example:
            df.filter(col("region") == "US").update({
                "backup_email": col("primary_email"),
                "score": col("base_score") * 1.5
            })
        """
        # Converti filtri API in formato storage
        storage_filters = []
        for f in self._filters:
            if isinstance(f, BinaryFilter) and isinstance(f.left, Column):
                storage_filters.append((f.left.name, f.op, f.right))
        
        # Compila vector ops se presenti
        vector_ops = self._compile_ops() if self._transformations else None
        
        # Esegui update
        self._storage.update(
            filters=storage_filters,
            updates=updates,
            vector_ops=vector_ops,
            operation="update"
        )
        
        return self


class DDIMSession:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)

    def read(self, table_name: str) -> DDIMFrame:
        path = self.base_path / table_name
        if not path.exists():
            raise FileNotFoundError(f"Dataset {table_name} not found")
        return DDIMFrame(NDimStorage(str(path)), table_name)

    def create_dataset(self, name: str, **kwargs) -> DDIMFrame:
        path = self.base_path / name
        return DDIMFrame(NDimStorage(str(path), **kwargs), name)
