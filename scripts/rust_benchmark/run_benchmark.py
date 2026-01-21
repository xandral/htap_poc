#!/usr/bin/env python3
"""
Benchmark Runner - Orchestrates the full benchmark pipeline.

Usage:
  # Single run with defaults
  python run_benchmark.py

  # Multiple configurations
  python run_benchmark.py --sweep

  # Custom config
  python run_benchmark.py --rows 500000 --chunk-rows 5000

  # From config file
  python run_benchmark.py --config my_config.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the generator
from generate_data import (
    DataConfig,
    create_duckdb_table,
    populate_duckdb,
    chunk_and_write,
    save_config,
    BATCH_SIZE,
)
import duckdb


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    # Data config
    num_rows: int = 10_000_000
    num_sensors: int = 200
    num_data_columns: int = 256
    chunk_rows: int = 100_000
    chunk_cols: int = 4
    seed: int = 42

    # Benchmark config
    benchmark_runs: int = 15
    warmup_runs: int = 5

    # Paths
    output_dir: str = "./benchmark_data"
    results_dir: str = "./results"

    # Metadata
    name: str = "default"
    description: str = ""

    def to_data_config(self) -> DataConfig:
        return DataConfig(
            num_rows=self.num_rows,
            num_sensors=self.num_sensors,
            num_data_columns=self.num_data_columns,
            chunk_rows=self.chunk_rows,
            chunk_cols=self.chunk_cols,
            seed=self.seed,
            output_dir=self.output_dir,
        )


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    timestamp: str
    generation_time_s: float
    duckdb_load_time_s: float
    duckdb_ram_mb: float
    total_files: int
    total_rows: int
    queries: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': asdict(self.config),
            'timestamp': self.timestamp,
            'generation_time_s': self.generation_time_s,
            'duckdb_load_time_s': self.duckdb_load_time_s,
            'duckdb_ram_mb': self.duckdb_ram_mb,
            'total_files': self.total_files,
            'total_rows': self.total_rows,
            'queries': self.queries,
        }


def update_rust_config(config: BenchmarkConfig, rust_src: Path):
    """No longer needed - Rust now reads config.json directly."""
    print(f"Rust will read config from {config.output_dir}/config.json")


def build_rust(rust_src: Path) -> bool:
    """Build Rust benchmark."""
    print("\nBuilding Rust benchmark...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=rust_src,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return False

    print("Build successful")
    return True


def run_rust_benchmark(rust_src: Path, data_dir: str, output_json: str) -> Optional[Dict]:
    """Run Rust benchmark and return results."""
    binary = rust_src / "target" / "release" / "parquet_benchmark"

    if not binary.exists():
        print(f"Binary not found: {binary}")
        return None

    print(f"\nRunning benchmark on {data_dir}...")
    result = subprocess.run(
        [str(binary), data_dir, output_json],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}")
        return None

    # Load results
    if Path(output_json).exists():
        with open(output_json) as f:
            return json.load(f)

    return None


def run_single_benchmark(config: BenchmarkConfig, rust_src: Path) -> Optional[BenchmarkResult]:
    """Run a single benchmark with given configuration."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {config.name}")
    print("=" * 70)
    print(f"Config: {config.num_rows:,} rows, {config.chunk_rows:,} chunk_rows, {config.num_sensors} sensors")

    # Setup directories
    data_dir = Path(config.output_dir)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json = results_dir / f"result_{config.name}_{timestamp}.json"

    # 1. Generate data
    print("\n--- Step 1: Generate Data ---")
    t0 = time.time()

    data_config = config.to_data_config()

    # Clean output directory
    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    # Create DuckDB and populate in batches
    db_path = data_dir / "benchmark.duckdb"
    conn = duckdb.connect(str(db_path))
    create_duckdb_table(conn, data_config)
    populate_duckdb(conn, data_config, BATCH_SIZE)

    print("Riordino fisico dei dati per simulare il partitioning...")
    conn.execute("CREATE TABLE iot_ordered AS SELECT * FROM iot ORDER BY sensor_id, timestamp")
    conn.execute("DROP TABLE iot")
    conn.execute("ALTER TABLE iot_ordered RENAME TO iot")

    row_count = conn.execute("SELECT COUNT(*) FROM iot").fetchone()[0]
    print(f"DuckDB populated: {row_count:,} rows")

    # Export chunks for Rust
    files, total_bytes = chunk_and_write(conn, data_config)
    save_config(data_config)
    conn.close()

    gen_time = time.time() - t0
    print(f"Generated {files} files ({total_bytes/(1024*1024):.1f} MB) in {gen_time:.2f}s")

    # 2. Update and build Rust
    print("\n--- Step 2: Build Rust ---")
    update_rust_config(config, rust_src)

    if not build_rust(rust_src):
        return None

    # 3. Run benchmark
    print("\n--- Step 3: Run Benchmark ---")
    results = run_rust_benchmark(rust_src, str(data_dir), str(result_json))

    if results is None:
        return None

    # 4. Create result object
    return BenchmarkResult(
        config=config,
        timestamp=timestamp,
        generation_time_s=gen_time,
        duckdb_load_time_s=results.get('duckdb_load_time_s', 0),
        duckdb_ram_mb=results.get('duckdb_ram_mb', 0),
        total_files=results.get('total_files', files),
        total_rows=results.get('total_rows', config.num_rows),
        queries=results.get('queries', []),
    )


def run_sweep(rust_src: Path, results_dir: Path, quick: bool = False) -> List[BenchmarkResult]:
    """Run benchmark sweep across multiple configurations."""

    if quick:
        # Quick sweep for testing
        configs = [
            BenchmarkConfig(
                name="quick_1k_chunks",
                description="Quick test: 1K chunks",
                num_rows=100_000,
                chunk_rows=1_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="quick_10k_chunks",
                description="Quick test: 10K chunks",
                num_rows=100_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
        ]
    else:
        # Full sweep configurations
        configs = [
            # === Vary chunk size ===
            BenchmarkConfig(
                name="chunk_1k",
                description="Small chunks (1K rows)",
                num_rows=1_000_000,
                chunk_rows=1_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="chunk_10k",
                description="Medium chunks (10K rows)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="chunk_50k",
                description="Large chunks (50K rows)",
                num_rows=1_000_000,
                chunk_rows=50_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),

            # === Vary number of sensors (hash buckets) ===
            BenchmarkConfig(
                name="sensors_3",
                description="3 sensors (hash buckets)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=3,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="sensors_10",
                description="10 sensors (hash buckets)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=10,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),

            # === Vary data size (rows) ===
            BenchmarkConfig(
                name="rows_500k",
                description="500K rows",
                num_rows=500_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="rows_2m",
                description="2M rows",
                num_rows=2_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=10,
                results_dir=str(results_dir),
            ),

            # === Vary number of columns (width) ===
            BenchmarkConfig(
                name="cols_5",
                description="Narrow table (5 columns)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=5,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="cols_20",
                description="Medium table (20 columns)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=20,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="cols_50",
                description="Wide table (50 columns)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=50,
                results_dir=str(results_dir),
            ),
            BenchmarkConfig(
                name="cols_100",
                description="Very wide table (100 columns)",
                num_rows=1_000_000,
                chunk_rows=10_000,
                chunk_cols=1,
                num_sensors=5,
                num_data_columns=100,
                results_dir=str(results_dir),
            ),
        ]

    results = []
    for i, config in enumerate(configs):
        print(f"\n{'#' * 70}")
        print(f"# SWEEP {i+1}/{len(configs)}: {config.name}")
        print(f"{'#' * 70}")

        result = run_single_benchmark(config, rust_src)
        if result:
            results.append(result)

            # Save intermediate results
            save_sweep_results(results, results_dir / "sweep_results.json")

    return results


def save_sweep_results(results: List[BenchmarkResult], path: Path):
    """Save sweep results to JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_configs': len(results),
        'results': [r.to_dict() for r in results],
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved sweep results to {path}")


def generate_sweep_report(results: List[BenchmarkResult], output_dir: Path):
    """Generate comparison report and charts for sweep results."""

    print("\n" + "=" * 70)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 70)

    # Summary table
    print(f"\n{'Config':<20} {'Rows':<12} {'ChunkRows':<12} {'Sensors':<10} {'Files':<8} {'DuckDB RAM':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r.config.name:<20} {r.config.num_rows:<12,} {r.config.chunk_rows:<12,} "
              f"{r.config.num_sensors:<10} {r.total_files:<8} {r.duckdb_ram_mb:<12.0f} MB")

    # Query comparison
    print(f"\n{'Config':<15}", end="")
    if results and results[0].queries:
        for q in results[0].queries[:4]:  # First 4 queries
            print(f" {q['name'][:12]:<14}", end="")
    print()
    print("-" * 80)

    for r in results:
        print(f"{r.config.name:<15}", end="")
        for q in r.queries[:4]:
            winner = "R" if q['winner'] == 'Rust' else "D" if q['winner'] == 'DuckDB' else "T"
            speedup = q['speedup'] if q['speedup'] > 1 else 1/q['speedup']
            print(f" {winner}:{speedup:<5.1f}x     ", end="")
        print()

    # Save detailed report
    report_path = output_dir / "sweep_report.txt"
    with open(report_path, 'w') as f:
        f.write("BENCHMARK SWEEP REPORT\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"\n{r.config.name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Rows: {r.config.num_rows:,}\n")
            f.write(f"  Chunk rows: {r.config.chunk_rows:,}\n")
            f.write(f"  Sensors: {r.config.num_sensors}\n")
            f.write(f"  Files: {r.total_files}\n")
            f.write(f"  DuckDB RAM: {r.duckdb_ram_mb:.0f} MB\n")
            f.write(f"  Gen time: {r.generation_time_s:.2f}s\n\n")

            f.write(f"  {'Query':<35} {'Rust p50':<12} {'Duck p50':<12} {'Winner':<10}\n")
            for q in r.queries:
                f.write(f"  {q['name']:<35} {q['rust_p50_ms']:<12.2f} {q['duckdb_p50_ms']:<12.2f} {q['winner']:<10}\n")
            f.write("\n")

    print(f"\nDetailed report saved to {report_path}")


def plot_sweep_results(results: List[BenchmarkResult], output_dir: Path):
    """Generate comparison plots for sweep results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Plot 1: Speedup by configuration
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Query names from first result
    query_names = [q['name'] for q in results[0].queries]
    config_names = [r.config.name for r in results]

    # Speedup heatmap
    ax = axes[0, 0]
    speedups = np.array([[q['speedup'] for q in r.queries] for r in results])
    im = ax.imshow(speedups, cmap='RdBu', aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(range(len(query_names)))
    ax.set_xticklabels([q[:10] for q in query_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=9)
    ax.set_title('Speedup by Config (>1 = Rust faster)', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # RAM usage comparison
    ax = axes[0, 1]
    rams = [r.duckdb_ram_mb for r in results]
    bars = ax.bar(config_names, rams, color='steelblue')
    ax.set_ylabel('DuckDB RAM (MB)')
    ax.set_title('Memory Usage by Config', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, ram in zip(bars, rams):
        ax.annotate(f'{ram:.0f}', xy=(bar.get_x() + bar.get_width()/2, ram),
                    ha='center', va='bottom', fontsize=8)

    # Files generated
    ax = axes[1, 0]
    files = [r.total_files for r in results]
    ax.bar(config_names, files, color='coral')
    ax.set_ylabel('Number of Parquet Files')
    ax.set_title('Files Generated by Config', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Best query speedup per config
    ax = axes[1, 1]
    best_speedups = [max(q['speedup'] for q in r.queries) for r in results]
    worst_speedups = [min(q['speedup'] for q in r.queries) for r in results]
    x = np.arange(len(config_names))
    ax.bar(x - 0.2, best_speedups, 0.4, label='Best (max pruning)', color='green', alpha=0.7)
    ax.bar(x + 0.2, worst_speedups, 0.4, label='Worst (min pruning)', color='red', alpha=0.7)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Speedup')
    ax.set_title('Best/Worst Speedup by Config', fontweight='bold')
    ax.legend()

    plt.suptitle('Benchmark Sweep Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved sweep comparison plot to {output_dir / 'sweep_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run partition pruning benchmark')

    # Mode
    parser.add_argument('--sweep', action='store_true', help='Run sweep across multiple configs')

    # Single run config
    parser.add_argument('--rows', type=int, default=1_000_000, help='Number of rows')
    parser.add_argument('--sensors', type=int, default=5, help='Number of sensors')
    parser.add_argument('--chunk-rows', type=int, default=10_000, help='Rows per chunk')
    parser.add_argument('--chunk-cols', type=int, default=4, help='Column groups')
    parser.add_argument('--data-columns', type=int, default=10, help='Metric columns')
    parser.add_argument('--benchmark-runs', type=int, default=15, help='Benchmark iterations')
    parser.add_argument('--warmup-runs', type=int, default=5, help='Warmup iterations')

    # Paths
    parser.add_argument('--output', type=str, default='./benchmark_data', help='Data output dir')
    parser.add_argument('--results', type=str, default='./results', help='Results output dir')
    parser.add_argument('--config', type=str, help='Load config from JSON')

    # Options
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--name', type=str, default='custom', help='Benchmark name')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        # Run sweep
        results = run_sweep(script_dir, results_dir)

        if results:
            generate_sweep_report(results, results_dir)
            if not args.no_plots:
                plot_sweep_results(results, results_dir)
    else:
        # Single run
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
            config = BenchmarkConfig(**config_dict)
        else:
            config = BenchmarkConfig(
                name=args.name,
                num_rows=args.rows,
                num_sensors=args.sensors,
                num_data_columns=args.data_columns,
                chunk_rows=args.chunk_rows,
                chunk_cols=args.chunk_cols,
                benchmark_runs=args.benchmark_runs,
                warmup_runs=args.warmup_runs,
                output_dir=args.output,
                results_dir=args.results,
            )

        result = run_single_benchmark(config, script_dir)

        if result and not args.no_plots:
            # Run standard plots
            try:
                subprocess.run([sys.executable, "plot_results.py"], cwd=script_dir)
            except Exception as e:
                print(f"Plot generation failed: {e}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
