
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import inspect
import re
from typing import List, Tuple, Union, Optional, Callable, Dict, Any


# ==========================================
# CORE VECTOR OPERATIONS
# ==========================================

class VectorOps:
    """Unified vector operations engine with generic N-D support."""

    # ==========================================
    # ML OPERATIONS (Matrix view when beneficial)
    # ==========================================

    @staticmethod
    def cosine_similarity_query(table: pa.Table, col: str, 
                                query_vector: List[float], 
                                result_col: str = "cosine_sim") -> pa.Table:
        """
        Compute cosine similarity against a query vector.
        Uses matrix view ONLY if data is 1D fixed-length vectors.
        """
        q = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(q)

        if norm_q == 0:
            return table.append_column(result_col, pa.array([0.0] * len(table)))

        # Try matrix view for performance (only on 1D vectors)
        mat = VectorOps._try_matrix_view_1d(table, col, expected_width=len(q))
        
        if mat is not None:
            # Fast vectorized path
            dot = np.dot(mat, q)
            norm_a = np.linalg.norm(mat, axis=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                sim = dot / (norm_a * norm_q)
                sim = np.nan_to_num(sim, nan=0.0)
            
            return table.append_column(result_col, pa.array(sim))
        
        # Generic fallback (row-by-row)
        chunked = table[col]
        if len(chunked) == 0:
            return table.append_column(result_col, pa.array([], type=pa.float64()))

        rows = chunked.to_pylist()
        sims = []
        
        for row in rows:
            if not row:
                sims.append(0.0)
                continue
            
            rv = np.array(row, dtype=np.float32).flatten()  # Handle any shape
            d = np.dot(rv, q)
            n = np.linalg.norm(rv)
            
            sims.append(float(d / (n * norm_q)) if n * norm_q > 0 else 0.0)
        
        return table.append_column(result_col, pa.array(sims))

    @staticmethod
    def cosine_similarity(table: pa.Table, cols_a: List[str], cols_b: List[str], 
                         result_col: str = "cosine_sim") -> pa.Table:
        """
        Cosine similarity between two column sets.
        Uses matrix view ONLY for single 1D vector columns.
        """
        # Try fast path: single columns with 1D vectors
        if len(cols_a) == 1 and len(cols_b) == 1:
            mat_a = VectorOps._try_matrix_view_1d(table, cols_a[0])
            mat_b = VectorOps._try_matrix_view_1d(table, cols_b[0])
            
            if mat_a is not None and mat_b is not None and mat_a.shape[1] == mat_b.shape[1]:
                # Fast vectorized computation
                dot = np.einsum('ij,ij->i', mat_a, mat_b)
                norm_a = np.linalg.norm(mat_a, axis=1)
                norm_b = np.linalg.norm(mat_b, axis=1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    sim = dot / (norm_a * norm_b)
                    sim = np.nan_to_num(sim, nan=0.0)
                
                return table.append_column(result_col, pa.array(sim))

        # Generic Arrow-native path
        col0_type = table.schema.field(cols_a[0]).type
        is_list = pa.types.is_list(col0_type) or pa.types.is_fixed_size_list(col0_type)

        if is_list:
            dot_product, norm_a, norm_b = VectorOps._compute_list_cosine(table, cols_a, cols_b)
        else:
            dot_product, norm_a, norm_b = VectorOps._compute_scalar_cosine(table, cols_a, cols_b)

        # Safe division
        denominator = pc.multiply(norm_a, norm_b)
        valid_mask = pc.greater(denominator, 0.0).fill_null(False)
        safe_denom = pc.if_else(valid_mask, denominator, 1.0)
        raw_sim = pc.divide(dot_product, safe_denom)
        sim_array = pc.if_else(valid_mask, raw_sim, 0.0)
        
        return table.append_column(result_col, sim_array)

    @staticmethod
    def _compute_list_cosine(table: pa.Table, cols_a: List[str], cols_b: List[str]) -> Tuple:
        """Compute cosine components for list columns (Arrow native)."""
        arr_a = table[cols_a[0]].combine_chunks()
        arr_b = table[cols_b[0]].combine_chunks()
        
        flat_a = arr_a.flatten()
        flat_b = arr_b.flatten()
        flat_mul = pc.multiply(flat_a, flat_b)
        
        lengths = arr_a.value_lengths().to_numpy()
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        
        if len(table) > 0:
            dot_np = np.add.reduceat(flat_mul.to_numpy(), offsets[:-1])
            dot_product = pa.array(dot_np)
        else:
            dot_product = pa.array([])

        def calc_norm(arr):
            flat = arr.flatten()
            sq = pc.multiply(flat, flat)
            lns = arr.value_lengths().to_numpy()
            offs = np.concatenate(([0], np.cumsum(lns))).astype(np.int32)
            if len(arr) == 0:
                return pa.array([])
            sum_sq = np.add.reduceat(sq.to_numpy(), offs[:-1])
            return pc.sqrt(pa.array(sum_sq))

        norm_a = calc_norm(arr_a)
        norm_b = calc_norm(arr_b)
        
        return dot_product, norm_a, norm_b

    @staticmethod
    def _compute_scalar_cosine(table: pa.Table, cols_a: List[str], cols_b: List[str]) -> Tuple:
        """Compute cosine components for scalar columns."""
        dot_product = pc.multiply(table[cols_a[0]], table[cols_b[0]])
        for ca, cb in zip(cols_a[1:], cols_b[1:]):
            dot_product = pc.add(dot_product, pc.multiply(table[ca], table[cb]))
        
        def compute_norm_sq(cols):
            acc = pc.multiply(table[cols[0]], table[cols[0]])
            for c in cols[1:]:
                acc = pc.add(acc, pc.multiply(table[c], table[c]))
            return pc.sqrt(acc)

        norm_a = compute_norm_sq(cols_a)
        norm_b = compute_norm_sq(cols_b)
        
        return dot_product, norm_a, norm_b

    # ==========================================
    # SLICING OPERATIONS (Generic N-D)
    # ==========================================

    @staticmethod
    def mda_slice_tensor(table: pa.Table, col: str, 
                         slices: List[Union[int, slice, Tuple[int, int]]], 
                         shape: Optional[Tuple[int, ...]] = None,
                         result_col: Optional[str] = None) -> pa.Table:
        """
        Generic N-D tensor slicing.
        
        1D case (no shape or shape=(n,)): Pure Arrow (fast)
        N-D case (shape provided): NumPy reshape (flexible)
        
        No forced 2D conversions - works with any dimensionality.
        """
        if result_col is None:
            is_1d = len(slices) == 1 and (shape is None or len(shape) == 1)
            result_col = f"{col}_slice_1d" if is_1d else f"{col}_slice_nd"
        
        # Fast path: 1D slicing (pure Arrow)
        if len(slices) == 1 and (shape is None or len(shape) == 1):
            s = slices[0]
            start = s.start if isinstance(s, slice) else s[0]
            end = s.stop if isinstance(s, slice) else s[1]
            sliced_arr = pc.list_slice(table[col], start, end)
            return table.append_column(result_col, sliced_arr)

        # N-D slicing: requires shape
        if shape is None:
            raise ValueError("Shape is required for N-D slicing (len(slices) > 1 or len(shape) > 1)")
        
        return VectorOps._slice_nd_generic(table, col, slices, shape, result_col)

    @staticmethod
    def _slice_nd_generic(table: pa.Table, col: str, slices: List, 
                         shape: Tuple, result_col: str) -> pa.Table:
        """
        Generic N-D tensor slicing using NumPy.
        Supports any dimensionality (2D, 3D, 4D, etc.).
        """
        arr = table[col].combine_chunks()
        flat_values = arr.flatten().to_numpy(zero_copy_only=False)
        
        # Validate shape
        total_elements = int(np.prod(shape))
        expected_length = len(arr) * total_elements
        
        if len(flat_values) != expected_length:
            raise ValueError(
                f"Shape mismatch: {len(flat_values)} elements != "
                f"{len(arr)} rows * {shape} = {expected_length}"
            )

        # Reshape to (n_rows, *shape)
        full_shape = (len(arr),) + tuple(shape)
        tensor = flat_values.reshape(full_shape)

        # Convert slice specifications to slice objects
        obj_slices = [slice(None)]  # Keep all rows
        
        for s in slices:
            if isinstance(s, (tuple, list)) and len(s) == 2:
                obj_slices.append(slice(s[0], s[1]))
            elif isinstance(s, slice):
                obj_slices.append(s)
            elif isinstance(s, int):
                obj_slices.append(s)
            else:
                raise ValueError(f"Invalid slice format: {s}")

        # Apply slicing
        sliced_tensor = tensor[tuple(obj_slices)]
        
        # Compute new shape per row
        result_shape = sliced_tensor.shape
        elements_per_row = int(np.prod(result_shape[1:]))
        
        # Flatten back to list format
        result_flat = sliced_tensor.reshape(-1)
        offsets = np.arange(0, (len(table) + 1) * elements_per_row, 
                          elements_per_row, dtype=np.int32)
        new_list_arr = pa.ListArray.from_arrays(offsets, result_flat)
        
        return table.append_column(result_col, new_list_arr)

    # ==========================================
    # FILTER OPERATIONS
    # ==========================================

    @staticmethod
    def mda_filter_by_index(table: pa.Table, col: str, index: int, 
                            operator: str, threshold: Union[float, Tuple[float, float]]) -> pa.Table:
        """
        Filter rows based on value at specific index.
        Uses matrix view only for 1D vectors as optimization.
        """
        op = operator.upper()
        
        # Try fast path: 1D vector with matrix view
        mat = VectorOps._try_matrix_view_1d(table, col)
        if mat is not None and index < mat.shape[1]:
            col_view = mat[:, index]
            mask = VectorOps._apply_comparison(col_view, op, threshold)
            return table.filter(pa.array(mask))

        # Generic Arrow path
        elm = pc.list_element(table[col], index)
        mask = VectorOps._build_comparison_mask(elm, op, threshold)
        return table.filter(mask.fill_null(False))

    @staticmethod
    def mda_filter_range(table: pa.Table, col: str, start: int, end: int, 
                        operator: str, threshold: Union[float, Tuple[float, float]], 
                        quantifier: str = 'ANY') -> pa.Table:
        """
        Filter based on a slice (sub-array).
        Generic implementation - works with any dimensionality.
        """
        sliced_col = pc.list_slice(table[col], start, end)
        sliced_pylist = sliced_col.to_pylist()
        
        op = operator.upper()
        q = quantifier.upper()
        mask = []
        
        for row in sliced_pylist:
            if not row:
                mask.append(False)
                continue
            
            # Flatten in case of nested structure
            flat_row = np.array(row).flatten() if isinstance(row[0], (list, tuple)) else row
            result = VectorOps._evaluate_range_condition(flat_row, op, threshold, q)
            mask.append(result)

        return table.filter(pa.array(mask))

    @staticmethod
    def _evaluate_range_condition(row: Union[List, np.ndarray], op: str, 
                                  threshold, quantifier: str) -> bool:
        """Evaluate range filter condition on flattened data."""
        if op == 'BETWEEN':
            lower, upper = threshold
            if quantifier == 'ANY':
                return any(lower <= x <= upper for x in row)
            else:
                return all(lower <= x <= upper for x in row)
        
        # Determine representative value
        if (op in ['>', '>='] and quantifier == 'ANY') or \
           (op in ['<', '<='] and quantifier == 'ALL'):
            check_val = max(row)
        elif (op in ['>', '>='] and quantifier == 'ALL') or \
             (op in ['<', '<='] and quantifier == 'ANY'):
            check_val = min(row)
        else:
            check_val = max(row) if quantifier == 'ANY' else min(row)

        return VectorOps._apply_scalar_comparison(check_val, op, threshold)

    # ==========================================
    # REDUCTION OPERATIONS
    # ==========================================

    @staticmethod
    def mda_reduce_mean(table: pa.Table, col: str, result_col: str = None) -> pa.Table:
        """
        Reduce: AVG(col).
        Uses matrix view only for 1D vectors as optimization.
        Generic fallback for N-D tensors.
        """
        if result_col is None:
            result_col = f"{col}_mean"
        
        # Try fast path: 1D vectors
        mat = VectorOps._try_matrix_view_1d(table, col)
        if mat is not None:
            means = np.mean(mat, axis=1)
            return table.append_column(result_col, pa.array(means))

        # Generic reduceat path (works for any shape)
        chunked_arr = table[col]
        arr = chunked_arr.combine_chunks()
        flat_values = arr.flatten().to_numpy()
        
        lengths = arr.value_lengths().to_numpy()
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        
        if len(table) == 0:
            return table
        
        sums = np.add.reduceat(flat_values, offsets[:-1])
        counts = lengths
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), 
                         where=counts!=0)
        
        return table.append_column(result_col, pa.array(means))

    @staticmethod
    def mda_cardinality(table: pa.Table, col: str, result_col: str = None) -> pa.Table:
        """Get array length (works for any shape)."""
        if result_col is None:
            result_col = f"{col}_len"
        return table.append_column(result_col, pc.list_value_length(table[col]))

    # ==========================================
    # ELEMENT-WISE OPERATIONS
    # ==========================================

    @staticmethod
    def mda_element_transform(table: pa.Table, col: str, op: str, value: float = None,
                             result_col: str = None) -> pa.Table:
        """
        Perform math operations on ALL elements inside arrays.
        Generic - works with any dimensionality.
        """
        if result_col is None:
            result_col = f"{col}_{op}"
        
        chunked_arr = table[col]
        arr = chunked_arr.combine_chunks()
        flat_values = arr.flatten()
        
        if op == "add":
            res = pc.add(flat_values, value)
        elif op == "mul":
            res = pc.multiply(flat_values, value)
        elif op == "log":
            res = pc.ln(flat_values)
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        new_list_arr = pa.ListArray.from_arrays(arr.offsets, res)
        return table.append_column(result_col, new_list_arr)

    # ==========================================
    # STRUCTURAL OPERATIONS
    # ==========================================

    @staticmethod
    def mda_unnest(table: pa.Table, col: str) -> pa.Table:
        """Unnest array column."""
        df = table.to_pandas().explode(col)
        return pa.Table.from_pandas(df)
    
    @staticmethod
    def mda_concat(table: pa.Table, col1: str, col2: str, result_col: str = None) -> pa.Table:
        """Concatenate two array columns."""
        if result_col is None:
            result_col = f"{col1}_cat_{col2}"
        
        a1 = table[col1].to_pylist()
        a2 = table[col2].to_pylist()
        res = [(l1 or []) + (l2 or []) for l1, l2 in zip(a1, a2)]
        
        return table.append_column(result_col, pa.array(res))

    # ==========================================
    # SCALAR OPERATIONS
    # ==========================================

    @staticmethod
    def scalar_op(table: pa.Table, col: str, op: str, scalar: float, 
                 result_col: str = None) -> pa.Table:
        """Scalar arithmetic operation."""
        if result_col is None:
            result_col = f"{col}_{op}_{scalar}"
        
        if op == "add":
            arr = pc.add(table[col], scalar)
        elif op == "sub":
            arr = pc.subtract(table[col], scalar)
        elif op == "mul":
            arr = pc.multiply(table[col], scalar)
        elif op == "div":
            arr = pc.divide(table[col], scalar)
        else:
            raise ValueError(f"Unknown scalar op: {op}")
        
        return table.append_column(result_col, arr)

    @staticmethod
    def element_wise_add(table: pa.Table, col1: str, col2: str, result_col: str = None) -> pa.Table:
        if result_col is None:
            result_col = f"{col1}_plus_{col2}"
        arr = pc.add(table[col1], table[col2])
        return table.append_column(result_col, arr)
    
    @staticmethod
    def element_wise_sub(table: pa.Table, col1: str, col2: str, result_col: str = None) -> pa.Table:
        if result_col is None:
            result_col = f"{col1}_minus_{col2}"
        arr = pc.subtract(table[col1], table[col2])
        return table.append_column(result_col, arr)
    
    @staticmethod
    def element_wise_mul(table: pa.Table, col1: str, col2: str, result_col: str = None) -> pa.Table:
        if result_col is None:
            result_col = f"{col1}_mul_{col2}"
        arr = pc.multiply(table[col1], table[col2])
        return table.append_column(result_col, arr)

    @staticmethod
    def math_n_sum(table: pa.Table, cols: List[str], result_col: str = None) -> pa.Table:
        if not cols:
            return table
        if result_col is None:
            result_col = "sum_" + "_".join(cols)
        
        acc = table[cols[0]]
        for col in cols[1:]:
            acc = pc.add(acc, table[col])
        
        return table.append_column(result_col, acc)

    @staticmethod
    def math_n_product(table: pa.Table, cols: List[str], result_col: str = None) -> pa.Table:
        if not cols:
            return table
        if result_col is None:
            result_col = "prod_" + "_".join(cols)
        
        acc = table[cols[0]]
        for col in cols[1:]:
            acc = pc.multiply(acc, table[col])
        
        return table.append_column(result_col, acc)

    @staticmethod
    def math_weighted_sum(table: pa.Table, cols: List[str], weights: List[float], 
                         result_col: str = None) -> pa.Table:
        if len(cols) != len(weights):
            raise ValueError("Mismatch in cols/weights")
        if result_col is None:
            result_col = "weighted_sum"
        
        acc = pc.multiply(table[cols[0]], weights[0])
        for col, w in zip(cols[1:], weights[1:]):
            acc = pc.add(acc, pc.multiply(table[col], w))
        
        return table.append_column(result_col, acc)

    # ==========================================
    # HELPER METHODS
    # ==========================================

    @staticmethod
    def _try_matrix_view_1d(table: pa.Table, col: str, 
                           expected_width: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Try to get a 2D matrix view for 1D FIXED-LENGTH vectors.
        
        USE CASE: Performance optimization for ML ops on uniform vectors.
        
        Returns None if:
        - Column doesn't exist
        - Not a list type
        - Variable length lists
        - Width doesn't match expected (if provided)
        
        This is ONLY used for performance, never for correctness.
        """
        if col not in table.column_names:
            return None
        
        chunked = table[col]
        arr = chunked.combine_chunks()
        
        if len(arr) == 0:
            return None

        # Only works for list types
        if not (pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type)):
            return None

        try:
            flat_vals = arr.flatten().to_numpy(zero_copy_only=False)
            
            # Determine width
            if pa.types.is_fixed_size_list(arr.type):
                width = arr.type.list_size
            else:
                # For variable lists, check if all have same length
                if len(arr) == 0:
                    return None
                width = arr.value_length(0).as_py()
            
            # Check expected width if provided
            if expected_width is not None and width != expected_width:
                return None
            
            # Validate uniform length
            if len(flat_vals) == len(arr) * width:
                return flat_vals.reshape(len(arr), width)
            
            return None
            
        except Exception:
            return None

    @staticmethod
    def _build_comparison_mask(arr: pa.Array, op: str, threshold) -> pa.Array:
        """Build comparison mask using Arrow compute."""
        if op == 'BETWEEN':
            return pc.and_(
                pc.greater_equal(arr, threshold[0]), 
                pc.less_equal(arr, threshold[1])
            )
        elif op == '>':
            return pc.greater(arr, threshold)
        elif op == '>=':
            return pc.greater_equal(arr, threshold)
        elif op == '<':
            return pc.less(arr, threshold)
        elif op == '<=':
            return pc.less_equal(arr, threshold)
        elif op == '==':
            return pc.equal(arr, threshold)
        elif op == '!=':
            return pc.not_equal(arr, threshold)
        else:
            raise ValueError(f"Unknown operator: {op}")

    @staticmethod
    def _apply_comparison(values: np.ndarray, op: str, threshold) -> np.ndarray:
        """Apply comparison operator to NumPy array."""
        if op == 'BETWEEN':
            return (values >= threshold[0]) & (values <= threshold[1])
        elif op == '>':
            return values > threshold
        elif op == '>=':
            return values >= threshold
        elif op == '<':
            return values < threshold
        elif op == '<=':
            return values <= threshold
        elif op == '==':
            return values == threshold
        elif op == '!=':
            return values != threshold
        else:
            return np.zeros(len(values), dtype=bool)

    @staticmethod
    def _apply_scalar_comparison(value: float, op: str, threshold) -> bool:
        """Apply comparison operator to scalar value."""
        if op == '>':
            return value > threshold
        elif op == '>=':
            return value >= threshold
        elif op == '<':
            return value < threshold
        elif op == '<=':
            return value <= threshold
        else:
            return False

class VectorOpSerializer:
    """
    Serializes and deserializes vector operations for patch log persistence.
    """

    # Registry of serializable operations
    _OPERATION_REGISTRY = {}

    @classmethod
    def register_operation(cls, name: str, func: Callable):
        """Register a named operation for serialization."""
        cls._OPERATION_REGISTRY[name] = func

    @staticmethod
    def serialize_ops(
        vector_ops: Optional[List[Callable]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Serialize a list of vector operations to JSON-compatible format.

        Returns:
            List of dicts with format:
            {
                "type": "registered" | "lambda" | "method",
                "name": str,
                "args": dict,
                "source": str (optional, for lambdas)
            }
        """
        if not vector_ops:
            return None

        serialized = []
        for op in vector_ops:
            try:
                serialized_op = VectorOpSerializer._serialize_single_op(op)
                serialized.append(serialized_op)
            except Exception as e:
                print(f"⚠️  Warning: Cannot serialize operation {op}: {e}")
                serialized.append(
                    {"type": "error", "error": str(e), "fallback": "skip"}
                )

        return serialized if serialized else None

    @staticmethod
    def _serialize_single_op(op: Callable) -> Dict[str, Any]:
        """Serialize a single operation."""

        # Case 1: Registered named operation
        for name, registered_func in VectorOpSerializer._OPERATION_REGISTRY.items():
            if op is registered_func or op.__name__ == registered_func.__name__:
                return {"type": "registered", "name": name}

        # Case 2: Lambda with extractable parameters
        if op.__name__ == "<lambda>":
            return VectorOpSerializer._serialize_lambda(op)

        # Case 3: Bound method (e.g., VectorOps.cosine_similarity)
        if hasattr(op, "__self__"):
            return {
                "type": "method",
                "class": op.__self__.__class__.__name__,
                "method": op.__name__,
            }

        # Case 4: Function reference
        if hasattr(op, "__module__") and hasattr(op, "__name__"):
            return {"type": "function", "module": op.__module__, "name": op.__name__}

        raise ValueError(f"Unsupported operation type: {type(op)}")

    @staticmethod
    def _serialize_lambda(lam: Callable) -> Dict[str, Any]:
        """
        Extract parameters from common lambda patterns.
        Supports patterns like:
        - lambda t: VectorOps.scalar_op(t, "col", "mul", 2.0)
        - lambda t: t.append_column("name", VectorOps.cosine_similarity(...))
        """
        import re

        source = inspect.getsource(lam).strip()

        # Pattern 1: VectorOps.method_name(t, arg1, arg2, ...)
        pattern1 = r"VectorOps\.(\w+)\(t,\s*([^)]+)\)"
        match1 = re.search(pattern1, source)
        if match1:
            method_name = match1.group(1)
            args_str = match1.group(2)

            # Parse arguments (simple eval for literals)
            args = {}
            try:
                # Split by comma, handle nested structures
                arg_parts = VectorOpSerializer._smart_split(args_str)

                # Common parameter names for each method
                param_map = {
                    "scalar_op": ["col", "op", "scalar", "result_col"],
                    "cosine_similarity": ["cols_a", "cols_b", "result_col"],
                    "element_wise_add": ["col1", "col2", "result_col"],
                    "mda_slice_tensor": ["col", "slices", "shape", "result_col"],
                    "mda_reduce_mean": ["col", "result_col"],
                    "mda_filter_by_index": ["col", "index", "operator", "threshold"],
                }

                if method_name in param_map:
                    param_names = param_map[method_name]
                    for i, arg in enumerate(arg_parts):
                        if i < len(param_names):
                            args[param_names[i]] = VectorOpSerializer._parse_arg(arg)
            except:
                pass

            return {
                "type": "lambda_vectorops",
                "method": method_name,
                "args": args,
                "source": source,  # Keep for debugging
            }

        # Pattern 2: t.append_column(name, expr)
        pattern2 = r't\.append_column\(["\'](\w+)["\']\s*,\s*(.+)\)'
        match2 = re.search(pattern2, source)
        if match2:
            col_name = match2.group(1)
            expr = match2.group(2)

            # Check if expression is a VectorOps call
            if "VectorOps" in expr:
                return {
                    "type": "lambda_append",
                    "result_col": col_name,
                    "expression": expr,
                    "source": source,
                }

        # Fallback: store source code (not executable, for reference only)
        return {
            "type": "lambda_source",
            "source": source,
            "warning": "Cannot execute, source code only",
        }

    @staticmethod
    def _smart_split(s: str) -> List[str]:
        """Split arguments respecting nested brackets and quotes."""
        parts = []
        current = []
        depth = 0
        in_string = False
        string_char = None

        for char in s:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif char in ("(", "[", "{") and not in_string:
                depth += 1
            elif char in (")", "]", "}") and not in_string:
                depth -= 1
            elif char == "," and depth == 0 and not in_string:
                parts.append("".join(current).strip())
                current = []
                continue

            current.append(char)

        if current:
            parts.append("".join(current).strip())

        return parts

    @staticmethod
    def _parse_arg(arg_str: str) -> Any:
        """Parse argument string to Python value."""
        arg_str = arg_str.strip()

        # String
        if arg_str.startswith(('"', "'")):
            return arg_str[1:-1]

        # Number
        try:
            if "." in arg_str:
                return float(arg_str)
            return int(arg_str)
        except:
            pass

        # List
        if arg_str.startswith("["):
            try:
                return eval(arg_str)
            except:
                pass

        # Fallback: return as string
        return arg_str

    @staticmethod
    def deserialize_ops(
        serialized_ops: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Callable]]:
        """
        Deserialize vector operations from JSON format.

        Returns:
            List of callable functions, or None if input is None
        """
        if not serialized_ops:
            return None

        from core.vector_operations import VectorOps

        ops = []
        for ser_op in serialized_ops:
            try:
                op_type = ser_op.get("type")

                if op_type == "error" or op_type == "lambda_source":
                    print(f" Skipping non-executable operation: {ser_op}")
                    continue

                if op_type == "registered":
                    name = ser_op["name"]
                    if name in VectorOpSerializer._OPERATION_REGISTRY:
                        ops.append(VectorOpSerializer._OPERATION_REGISTRY[name])
                    else:
                        print(f" Unknown registered operation: {name}")

                elif op_type == "lambda_vectorops":
                    method_name = ser_op["method"]
                    args = ser_op["args"]

                    # Reconstruct lambda
                    if hasattr(VectorOps, method_name):
                        method = getattr(VectorOps, method_name)

                        # Create lambda with proper argument binding
                        if method_name == "scalar_op":
                            op = lambda t, m=method, a=args: m(
                                t, a["col"], a["op"], a["scalar"], a.get("result_col")
                            )
                        elif method_name == "cosine_similarity":
                            op = lambda t, m=method, a=args: t.append_column(
                                a.get("result_col", "cosine_sim"),
                                m(t, a["cols_a"], a["cols_b"]),
                            )
                        elif method_name == "mda_slice_tensor":
                            op = lambda t, m=method, a=args: m(
                                t,
                                a["col"],
                                a["slices"],
                                a.get("shape"),
                                a.get("result_col"),
                            )
                        elif method_name == "mda_reduce_mean":
                            op = lambda t, m=method, a=args: m(
                                t, a["col"], a.get("result_col")
                            )
                        else:
                            print(
                                f"  Unsupported method for deserialization: {method_name}"
                            )
                            continue

                        ops.append(op)

                elif op_type == "function":
                    # Import and get function reference
                    import importlib

                    module = importlib.import_module(ser_op["module"])
                    func = getattr(module, ser_op["name"])
                    ops.append(func)

                else:
                    print(f"  Unknown operation type: {op_type}")

            except Exception as e:
                print(f" Error deserializing operation {ser_op}: {e}")
                continue

        return ops if ops else None
