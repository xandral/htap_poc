#!/usr/bin/env python3
"""
Test script for column clone updates and delete operations.
Run this after applying the modifications to verify everything works.
"""

import shutil
import sys
import numpy as np
import pyarrow as pa
from pathlib import Path

# Assuming modifications have been applied
from src.core.api import DDIMSession, col
from src.core.expressions import (
    ColumnRef, ArithmeticExpr, ExpressionEvaluator, serialize_updates
)

BASE_PATH = "./test_column_updates"


def setup():
    """Clean test directory."""
    p = Path(BASE_PATH)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def test_expression_serialization():
    """Test that expressions serialize/deserialize correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Expression Serialization")
    print("=" * 60)
    
    # Test ColumnRef
    ref = ColumnRef("price")
    serialized = ref.to_dict()
    assert serialized == {"type": "column_ref", "name": "price"}
    deserialized = ColumnRef.from_dict(serialized)
    assert deserialized.name == "price"
    print("  ✅ ColumnRef serialization OK")
    
    # Test ArithmeticExpr (simple)
    expr = ArithmeticExpr(ColumnRef("price"), "*", 2)
    serialized = expr.to_dict()
    assert serialized["type"] == "arithmetic_expr"
    assert serialized["op"] == "*"
    assert serialized["left"] == {"type": "column_ref", "name": "price"}
    assert serialized["right"] == {"type": "scalar", "value": 2}
    print("  ✅ Simple ArithmeticExpr serialization OK")
    
    # Test nested expression: price * (1 - discount)
    inner = ArithmeticExpr(1, "-", ColumnRef("discount"))
    outer = ArithmeticExpr(ColumnRef("price"), "*", inner)
    serialized = outer.to_dict()
    
    # Deserialize and verify structure
    restored = ArithmeticExpr.from_dict(serialized)
    assert isinstance(restored.left, ColumnRef)
    assert restored.left.name == "price"
    assert restored.op == "*"
    assert isinstance(restored.right, ArithmeticExpr)
    print("  ✅ Nested ArithmeticExpr serialization OK")
    
    print("\n✅ All serialization tests passed!")


def test_expression_evaluation():
    """Test that expressions evaluate correctly against tables."""
    print("\n" + "=" * 60)
    print("TEST 2: Expression Evaluation")
    print("=" * 60)
    
    # Create test table
    table = pa.Table.from_pydict({
        "price": [100.0, 200.0, 300.0],
        "discount": [0.1, 0.2, 0.3],
        "quantity": [1, 2, 3]
    })
    
    # Test column reference
    expr = {"type": "column_ref", "name": "price"}
    result = ExpressionEvaluator.evaluate(expr, table)
    assert result.to_pylist() == [100.0, 200.0, 300.0]
    print("  ✅ Column reference evaluation OK")
    
    # Test arithmetic: price * 2
    expr = {
        "type": "arithmetic_expr",
        "left": {"type": "column_ref", "name": "price"},
        "op": "*",
        "right": {"type": "scalar", "value": 2}
    }
    result = ExpressionEvaluator.evaluate(expr, table)
    assert result.to_pylist() == [200.0, 400.0, 600.0]
    print("  ✅ Simple arithmetic evaluation OK")
    
    # Test nested: price * (1 - discount)
    expr = {
        "type": "arithmetic_expr",
        "left": {"type": "column_ref", "name": "price"},
        "op": "*",
        "right": {
            "type": "arithmetic_expr",
            "left": {"type": "scalar", "value": 1},
            "op": "-",
            "right": {"type": "column_ref", "name": "discount"}
        }
    }
    result = ExpressionEvaluator.evaluate(expr, table)
    expected = [100*0.9, 200*0.8, 300*0.7]  # price * (1-discount)
    for i, (r, e) in enumerate(zip(result.to_pylist(), expected)):
        assert abs(r - e) < 0.001, f"Mismatch at {i}: {r} vs {e}"
    print("  ✅ Nested arithmetic evaluation OK")
    
    # Test get_required_columns
    columns = ExpressionEvaluator.get_required_columns(expr)
    assert columns == {"price", "discount"}
    print("  ✅ Required columns extraction OK")
    
    print("\n✅ All evaluation tests passed!")


def test_serialize_updates():
    """Test the serialize_updates helper function."""
    print("\n" + "=" * 60)
    print("TEST 3: Update Serialization")
    print("=" * 60)
    
    # Create a mock Column class (simulating api.py Column)
    class MockColumn:
        def __init__(self, name):
            self.name = name
    
    updates = {
        "scalar_field": 42,
        "string_field": "active",
        "column_ref": ColumnRef("other_col"),
        "mock_column": MockColumn("source_col"),
        "expression": ArithmeticExpr(ColumnRef("a"), "+", ColumnRef("b"))
    }
    
    serialized = serialize_updates(updates)
    
    assert serialized["scalar_field"] == 42
    assert serialized["string_field"] == "active"
    assert serialized["column_ref"] == {"type": "column_ref", "name": "other_col"}
    assert serialized["mock_column"] == {"type": "column_ref", "name": "source_col"}
    assert serialized["expression"]["type"] == "arithmetic_expr"
    
    print("  ✅ Scalar serialization OK")
    print("  ✅ ColumnRef serialization OK")
    print("  ✅ Column object serialization OK")
    print("  ✅ ArithmeticExpr serialization OK")
    
    print("\n✅ All update serialization tests passed!")


def test_column_arithmetic_operators():
    """Test that Column class arithmetic operators work correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Column Arithmetic Operators")
    print("=" * 60)
    
    # This tests the modifications to api.py Column class
    try:
        # These should create ArithmeticExpr objects
        expr1 = col("price") + 10
        expr2 = col("price") * col("quantity")
        expr3 = col("price") * (1 - col("discount"))
        
        # Verify types
        assert isinstance(expr1, ArithmeticExpr), f"Expected ArithmeticExpr, got {type(expr1)}"
        assert isinstance(expr2, ArithmeticExpr), f"Expected ArithmeticExpr, got {type(expr2)}"
        
        print("  ✅ col() + scalar creates ArithmeticExpr")
        print("  ✅ col() * col() creates ArithmeticExpr")
        
        # Test serialization
        serialized = expr1.to_dict()
        assert serialized["type"] == "arithmetic_expr"
        print("  ✅ Expression serializes correctly")
        
        print("\n✅ Column arithmetic operators work!")
        
    except TypeError as e:
        print(f"  ❌ Column arithmetic not implemented yet: {e}")
        print("     Apply the modifications to api.py Column class")
        return False
    
    return True


def test_full_integration():
    """Full integration test with storage."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Integration (Storage + Updates)")
    print("=" * 60)
    
    setup()
    session = DDIMSession(BASE_PATH)
    
    # Create dataset
    df = session.create_dataset("products", hash_dims={"category": 3})
    
    # Insert test data
    data = pa.Table.from_pydict({
        "category": ["A", "B", "A", "B", "A"],
        "price": [100.0, 200.0, 150.0, 250.0, 300.0],
        "discount": [0.1, 0.2, 0.15, 0.25, 0.3],
        "final_price": [0.0, 0.0, 0.0, 0.0, 0.0]  # Will be computed
    })
    df.write(data)
    print(f"  ✅ Inserted {len(data)} rows")
    
    
    # Arithmetic expression update
    print("\n--- Testing arithmetic expression update ---")
    try:
        # Update final_price = price * (1 - discount)
        df.update({
            "final_price": col("price") * (1 - col("discount"))
        })
        
        # Verify
        result = df.select("price", "discount", "final_price").collect()
        for p, d, fp in zip(
            result["price"].to_pylist(),
            result["discount"].to_pylist(),
            result["final_price"].to_pylist()
        ):
            expected = p * (1 - d)
            assert abs(fp - expected) < 0.01, f"Expected {expected}, got {fp}"
        print("  ✅ Arithmetic expression update works")
    except Exception as e:
        print(f"  ❌ Arithmetic expression update failed: {e}")
        return False
    
    # Delete operation
    print("\n--- Testing delete operation ---")
    try:
        initial_count = len(df.collect())
        
        df.filter(col("category") == "B").delete()
        
        remaining = df.select("category").collect()
        remaining_count = len(remaining)
        
        # Should have removed category B rows (2 rows)
        assert remaining_count == initial_count - 2, \
            f"Expected {initial_count - 2} rows, got {remaining_count}"
        
        # Verify no B categories remain
        categories = remaining["category"].to_pylist()
        assert "B" not in categories, f"Category B should be deleted, found in {categories}"
        print("  ✅ Delete operation works")
    except Exception as e:
        print(f"  ❌ Delete operation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("COLUMN UPDATE & DELETE TEST SUITE")
    print("=" * 60)
    
    # Unit tests (don't require storage modifications)
    test_expression_serialization()
    test_expression_evaluation()
    test_serialize_updates()
    
    # Tests that require api.py modifications
    if test_column_arithmetic_operators():
        # Full integration test (requires all modifications)
        test_full_integration()
    else:
        print("\n⚠️ Skipping integration tests - apply api.py modifications first")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()