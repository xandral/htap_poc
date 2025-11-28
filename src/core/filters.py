from typing import Any, List, Tuple, Union, Optional
import pyarrow as pa
import pyarrow.compute as pc

FilterCondition = Tuple[str, str, Any]
FilterExpression = List[Union[FilterCondition, str, list]]


class FilterEngine:
    """Evaluates complex logic expressions against PyArrow Tables."""

    @staticmethod
    def apply_op(column: pa.Array, op: str, value: Any):
        if op == "=":
            return pc.equal(column, value)
        if op == "!=":
            return pc.not_equal(column, value)
        if op == ">":
            return pc.greater(column, value)
        if op == ">=":
            return pc.greater_equal(column, value)
        if op == "<":
            return pc.less(column, value)
        if op == "<=":
            return pc.less_equal(column, value)
        if op == "IN":
            return pc.is_in(column, value_set=pa.array(value))
        raise ValueError(f"Unsupported operator: {op}")

    @classmethod
    def evaluate(
        cls, table: pa.Table, expression: FilterExpression
    ) -> Optional[pa.Array]:
        """
        Returns a Boolean mask. Returns None if columns are missing.
        """
        if not expression:
            return pa.array([True] * len(table))

        if isinstance(expression, tuple):
            col, op, val = expression
            if col not in table.column_names:
                return None
            return cls.apply_op(table[col], op, val)

        current_mask = None
        current_op = "AND"

        for item in expression:
            if isinstance(item, str):
                if item.upper() in ["AND", "OR"]:
                    current_op = item.upper()
                continue

            next_mask = cls.evaluate(table, item)

            if next_mask is None:
                return None

            if current_mask is None:
                current_mask = next_mask
            else:
                if current_op == "AND":
                    current_mask = pc.and_(current_mask, next_mask)
                elif current_op == "OR":
                    current_mask = pc.or_(current_mask, next_mask)

        return (
            current_mask if current_mask is not None else pa.array([True] * len(table))
        )

    @staticmethod
    def extract_columns(expression: FilterExpression) -> set:
        cols = set()
        if not expression:
            return cols
        if isinstance(expression, tuple):
            cols.add(expression[0])
            return cols
        if isinstance(expression, list):
            for item in expression:
                if isinstance(item, (list, tuple)):
                    cols.update(FilterEngine.extract_columns(item))
        return cols
