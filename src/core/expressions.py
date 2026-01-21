"""
Serializable expressions for update operations.
Separated to avoid circular imports between api.py and storage.py.

This module provides the foundation for "column clone" updates,
allowing expressions like:
    
    df.update({"col_a": col("col_b") * 2 + col("col_c")})

All expressions can be serialized to JSON for the patch log.
"""

from dataclasses import dataclass
from typing import Any, Union, Dict


@dataclass
class ColumnRef:
    """
    Serializable reference to a column.
    Used in updates to copy values from other columns.
    
    Example:
        ColumnRef("price")  # Reference to 'price' column
    """
    name: str
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {"type": "column_ref", "name": self.name}
    
    @staticmethod
    def from_dict(data: dict) -> "ColumnRef":
        """Deserialize from dict."""
        return ColumnRef(data["name"])
    
    def __repr__(self):
        return f"ColumnRef({self.name!r})"


@dataclass 
class ArithmeticExpr:
    """
    Serializable arithmetic expression for column updates.
    Supports: +, -, *, /
    
    These expressions form a tree that can be:
    1. Serialized to JSON for the patch log
    2. Evaluated at read-time against a PyArrow table
    
    Example:
        col("price") * (1 - col("discount"))
        
        Becomes:
        ArithmeticExpr(
            left=ColumnRef("price"),
            op="*",
            right=ArithmeticExpr(
                left=1,
                op="-", 
                right=ColumnRef("discount")
            )
        )
    """
    left: Any   # ColumnRef, ArithmeticExpr, or scalar (int, float, str)
    op: str     # "+", "-", "*", "/"
    right: Any
    
    def to_dict(self) -> dict:
        """Serialize expression tree to JSON-compatible dict."""
        return {
            "type": "arithmetic_expr",
            "left": self._serialize_operand(self.left),
            "op": self.op,
            "right": self._serialize_operand(self.right)
        }
    
    def _serialize_operand(self, operand) -> dict:
        """Serialize a single operand (left or right)."""
        if isinstance(operand, ColumnRef):
            return operand.to_dict()
        elif isinstance(operand, ArithmeticExpr):
            return operand.to_dict()
        elif hasattr(operand, 'name') and hasattr(operand, 'to_dict'):
            # ColumnRef-like object
            return {"type": "column_ref", "name": operand.name}
        elif hasattr(operand, 'name'):
            # Column object from api.py (has .name but no .to_dict)
            return {"type": "column_ref", "name": operand.name}
        else:
            # Scalar value
            return {"type": "scalar", "value": operand}
    
    @staticmethod
    def from_dict(data: dict) -> "ArithmeticExpr":
        """Deserialize expression tree from dict."""
        left = ArithmeticExpr._deserialize_operand(data["left"])
        right = ArithmeticExpr._deserialize_operand(data["right"])
        return ArithmeticExpr(left, data["op"], right)
    
    @staticmethod
    def _deserialize_operand(data):
        """Deserialize a single operand."""
        if not isinstance(data, dict):
            # Raw scalar value
            return data
        
        dtype = data.get("type")
        if dtype == "column_ref":
            return ColumnRef(data["name"])
        elif dtype == "arithmetic_expr":
            return ArithmeticExpr.from_dict(data)
        elif dtype == "scalar":
            return data["value"]
        else:
            # Unknown type, return as-is
            return data
    
    def __repr__(self):
        return f"ArithmeticExpr({self.left!r} {self.op} {self.right!r})"


class ExpressionEvaluator:
    """
    Evaluates serialized expressions against a PyArrow table.
    
    This is used during merge-on-read to compute update values
    from column references and arithmetic expressions.
    """
    
    @staticmethod
    def evaluate(expr, table) -> Any:
        """
        Evaluate an expression and return the result.
        
        Args:
            expr: Can be:
                - dict (serialized expression from patch log)
                - ColumnRef
                - ArithmeticExpr
                - scalar value
            table: PyArrow Table to evaluate against
            
        Returns:
            PyArrow Array or scalar value
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        
        # Case 1: Serialized dict from patch log
        if isinstance(expr, dict):
            expr_type = expr.get("type")
            
            if expr_type == "column_ref":
                col_name = expr["name"]
                if col_name in table.column_names:
                    return table[col_name].combine_chunks()
                else:
                    raise ValueError(f"Column '{col_name}' not found in table")
            
            elif expr_type == "arithmetic_expr":
                return ExpressionEvaluator._eval_arithmetic(expr, table)
            
            elif expr_type == "scalar":
                return expr["value"]
            
            elif expr_type == "callable":
                # Non-serialized lambda - can't evaluate
                raise ValueError("Cannot evaluate non-serialized callable from patch log")
            
            else:
                # Unknown type, return as-is
                return expr
        
        # Case 2: ColumnRef object
        elif isinstance(expr, ColumnRef):
            return table[expr.name].combine_chunks()
        
        # Case 3: ArithmeticExpr object
        elif isinstance(expr, ArithmeticExpr):
            return ExpressionEvaluator._eval_arithmetic(expr.to_dict(), table)
        
        # Case 4: Callable (runtime only)
        elif callable(expr):
            return expr(table)
        
        # Case 5: Object with .name attribute (Column from api.py)
        elif hasattr(expr, 'name'):
            return table[expr.name].combine_chunks()
        
        # Case 6: Scalar value
        else:
            return expr
    
    @staticmethod
    def _eval_arithmetic(expr_dict: dict, table) -> Any:
        """Recursively evaluate an arithmetic expression."""
        import pyarrow as pa
        import pyarrow.compute as pc
        
        left = expr_dict["left"]
        right = expr_dict["right"]
        op = expr_dict["op"]
        
        # Evaluate operands recursively
        left_val = ExpressionEvaluator.evaluate(left, table)
        right_val = ExpressionEvaluator.evaluate(right, table)
        
        # Convert scalars to arrays if needed for PyArrow compute
        if not isinstance(left_val, (pa.Array, pa.ChunkedArray)):
            left_val = pa.array([left_val] * len(table))
        if not isinstance(right_val, (pa.Array, pa.ChunkedArray)):
            right_val = pa.array([right_val] * len(table))
        
        # Apply operation
        ops_map = {
            "+": pc.add,
            "-": pc.subtract,
            "*": pc.multiply,
            "/": pc.divide
        }
        
        if op in ops_map:
            return ops_map[op](left_val, right_val)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    @staticmethod
    def get_required_columns(expr) -> set:
        """
        Extract all column names referenced in an expression.
        
        Used to determine which columns need to be loaded from storage.
        """
        columns = set()
        ExpressionEvaluator._extract_columns(expr, columns)
        return columns
    
    @staticmethod
    def _extract_columns(expr, columns: set):
        """Recursively extract column names."""
        if isinstance(expr, dict):
            expr_type = expr.get("type")
            if expr_type == "column_ref":
                columns.add(expr["name"])
            elif expr_type == "arithmetic_expr":
                ExpressionEvaluator._extract_columns(expr.get("left"), columns)
                ExpressionEvaluator._extract_columns(expr.get("right"), columns)
        
        elif isinstance(expr, ColumnRef):
            columns.add(expr.name)
        
        elif isinstance(expr, ArithmeticExpr):
            ExpressionEvaluator._extract_columns(expr.left, columns)
            ExpressionEvaluator._extract_columns(expr.right, columns)
        
        elif hasattr(expr, 'name'):
            columns.add(expr.name)


def serialize_updates(updates: Dict) -> Dict:
    """
    Serialize update values for the patch log.
    
    Handles:
    - Scalars: stored directly
    - Column objects: converted to column_ref
    - ColumnRef: serialized via to_dict()
    - ArithmeticExpr: serialized via to_dict()
    - Callables: marked as runtime-only with warning
    
    Args:
        updates: Dict of {column_name: update_value}
        
    Returns:
        Dict with serialized values suitable for JSON
    """
    serialized = {}
    
    for col_name, value in updates.items():
        if isinstance(value, ArithmeticExpr):
            serialized[col_name] = value.to_dict()
        
        elif isinstance(value, ColumnRef):
            serialized[col_name] = value.to_dict()
        
        elif hasattr(value, 'name') and hasattr(value, 'to_dict'):
            # ColumnRef-like
            serialized[col_name] = value.to_dict()
        
        elif hasattr(value, 'name'):
            # Column object from api.py
            serialized[col_name] = {"type": "column_ref", "name": value.name}
        
        elif callable(value):
            # Lambda: NOT serializable
            print(f"⚠️ Warning: callable for '{col_name}' will not be persisted in patch log")
            serialized[col_name] = {"type": "callable", "warning": "runtime_only"}
        
        else:
            # Scalar: store directly
            serialized[col_name] = value
    
    return serialized