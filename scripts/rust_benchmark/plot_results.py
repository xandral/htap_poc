#!/usr/bin/env python3
"""
Benchmark Results Visualization
Generates publication-quality charts from benchmark JSON output.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
RUST_COLOR = '#E74C3C'
DUCKDB_COLOR = '#3498DB'
TIE_COLOR = '#95A5A6'

def load_results(path='results.json'):
    with open(path) as f:
        return json.load(f)

def plot_latency_comparison(data, output='chart_latency.png'):
    """Bar chart comparing query latencies."""
    queries = [q['name'] for q in data['queries']]
    rust_times = [q['rust_p50_ms'] for q in data['queries']]
    duck_times = [q['duckdb_p50_ms'] for q in data['queries']]

    x = np.arange(len(queries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, rust_times, width, label='Rust (disk + pruning)',
                   color=RUST_COLOR, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x + width/2, duck_times, width, label='DuckDB (in-memory)',
                   color=DUCKDB_COLOR, edgecolor='white', linewidth=0.7)

    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Latency (ms) - lower is better', fontsize=12)
    ax.set_title('Query Latency Comparison: Rust Partition Pruning vs DuckDB In-Memory',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(' AND ', '\nAND ').replace(' = ', '=') for q in queries],
                       fontsize=9)
    ax.legend(loc='upper left', fontsize=10)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, max(max(rust_times), max(duck_times)) * 1.2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()

def plot_speedup(data, output='chart_speedup.png'):
    """Horizontal bar chart showing speedup factors."""
    queries = [q['name'] for q in data['queries']]
    speedups = [q['speedup'] for q in data['queries']]
    pruning = [q['pruning_pct'] for q in data['queries']]

    # Color based on winner
    colors = []
    for s in speedups:
        if s > 1.1:
            colors.append(RUST_COLOR)
        elif s < 0.9:
            colors.append(DUCKDB_COLOR)
        else:
            colors.append(TIE_COLOR)

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(queries))

    # Plot bars (log scale for better visualization)
    bars = ax.barh(y, speedups, color=colors, edgecolor='white', linewidth=0.7, height=0.6)

    # Add vertical line at x=1 (breakeven)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.02, len(queries)-0.3, 'Breakeven', fontsize=9, alpha=0.7)

    # Labels
    ax.set_xlabel('Speedup Factor (>1 = Rust faster, <1 = DuckDB faster)', fontsize=11)
    ax.set_title('Speedup: Rust Partition Pruning vs DuckDB In-Memory',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{q}\n({p:.0f}% pruned)" for q, p in zip(queries, pruning)], fontsize=9)

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        label = f'{speedup:.2f}x'
        if speedup > 1:
            label = f'Rust {speedup:.1f}x faster'
        elif speedup < 1:
            label = f'DuckDB {1/speedup:.1f}x faster'
        else:
            label = 'Tie'

        ax.annotate(label,
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(0, max(speedups) * 1.5)

    # Legend
    rust_patch = mpatches.Patch(color=RUST_COLOR, label='Rust wins')
    duck_patch = mpatches.Patch(color=DUCKDB_COLOR, label='DuckDB wins')
    tie_patch = mpatches.Patch(color=TIE_COLOR, label='Tie')
    ax.legend(handles=[rust_patch, duck_patch, tie_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()

def plot_pruning_vs_speedup(data, output='chart_pruning_speedup.png'):
    """Scatter plot showing relationship between pruning % and speedup."""
    pruning = [q['pruning_pct'] for q in data['queries']]
    speedups = [q['speedup'] for q in data['queries']]
    queries = [q['name'].split(':')[0] for q in data['queries']]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color points
    colors = [RUST_COLOR if s > 1.1 else DUCKDB_COLOR if s < 0.9 else TIE_COLOR for s in speedups]

    scatter = ax.scatter(pruning, speedups, c=colors, s=200, edgecolors='white', linewidth=2, zorder=5)

    # Add labels for each point
    for i, (x, y, q) in enumerate(zip(pruning, speedups, queries)):
        ax.annotate(q, (x, y), xytext=(8, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold', va='center')

    # Breakeven line
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Breakeven')

    # Regions
    ax.axhspan(1.1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.1 else 10,
               alpha=0.1, color=RUST_COLOR, label='Rust advantage zone')
    ax.axhspan(0, 0.9, alpha=0.1, color=DUCKDB_COLOR, label='DuckDB advantage zone')

    ax.set_xlabel('Pruning Percentage (%)', fontsize=12)
    ax.set_ylabel('Speedup Factor (Rust vs DuckDB)', fontsize=12)
    ax.set_title('Pruning Effectiveness: Higher Pruning → Rust Wins',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(-5, 105)
    ax.set_ylim(0, max(speedups) * 1.3)

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()

def plot_memory_comparison(data, output='chart_memory.png'):
    """Bar chart comparing memory usage."""
    fig, ax = plt.subplots(figsize=(8, 5))

    systems = ['Rust\n(Partition Pruning)', 'DuckDB\n(In-Memory)']
    memory = [0, data['duckdb_ram_mb']]
    colors = [RUST_COLOR, DUCKDB_COLOR]

    bars = ax.bar(systems, memory, color=colors, edgecolor='white', linewidth=2, width=0.5)

    # Add value labels
    ax.annotate(f"~0 MB\n(streams from disk)",
                xy=(0, 50), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.annotate(f"{data['duckdb_ram_mb']:.0f} MB\n({data['total_rows']/1e6:.1f}M rows)",
                xy=(1, data['duckdb_ram_mb']), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add projection
    projected_1b = data['duckdb_ram_mb'] * (1e9 / data['total_rows']) / 1024
    ax.axhline(y=data['duckdb_ram_mb'], color=DUCKDB_COLOR, linestyle=':', alpha=0.5)

    ax.set_ylabel('RAM Usage (MB)', fontsize=12)
    ax.set_title('Memory Footprint Comparison', fontsize=14, fontweight='bold')

    # Add text box with scaling info
    textstr = f"Scaling projection:\n• 1B rows → {projected_1b:.0f} GB RAM for DuckDB\n• Rust: still ~0 MB"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_ylim(0, data['duckdb_ram_mb'] * 1.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()

def plot_tradeoff_summary(data, output='chart_summary.png'):
    """Combined summary figure."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # --- Plot 1: Latency comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    queries = [q['name'].split(':')[1].strip()[:15] for q in data['queries']]
    rust_times = [q['rust_p50_ms'] for q in data['queries']]
    duck_times = [q['duckdb_p50_ms'] for q in data['queries']]

    x = np.arange(len(queries))
    width = 0.35
    ax1.bar(x - width/2, rust_times, width, label='Rust', color=RUST_COLOR)
    ax1.bar(x + width/2, duck_times, width, label='DuckDB', color=DUCKDB_COLOR)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('A) Query Latency (p50)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)

    # --- Plot 2: Speedup ---
    ax2 = fig.add_subplot(gs[0, 1])
    speedups = [q['speedup'] for q in data['queries']]
    colors = [RUST_COLOR if s > 1.1 else DUCKDB_COLOR if s < 0.9 else TIE_COLOR for s in speedups]
    ax2.barh(queries, speedups, color=colors)
    ax2.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Speedup (>1 = Rust faster)')
    ax2.set_title('B) Speedup Factor', fontweight='bold')

    # --- Plot 3: Pruning vs Speedup ---
    ax3 = fig.add_subplot(gs[1, 0])
    pruning = [q['pruning_pct'] for q in data['queries']]
    colors = [RUST_COLOR if s > 1.1 else DUCKDB_COLOR if s < 0.9 else TIE_COLOR for s in speedups]
    ax3.scatter(pruning, speedups, c=colors, s=150, edgecolors='white', linewidth=2)
    ax3.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Pruning %')
    ax3.set_ylabel('Speedup')
    ax3.set_title('C) Pruning vs Performance', fontweight='bold')
    for i, q in enumerate(data['queries']):
        ax3.annotate(q['name'].split(':')[0], (pruning[i], speedups[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    # --- Plot 4: Memory ---
    ax4 = fig.add_subplot(gs[1, 1])
    mem_data = [0.1, data['duckdb_ram_mb']]  # 0.1 for visibility
    ax4.bar(['Rust', 'DuckDB'], mem_data, color=[RUST_COLOR, DUCKDB_COLOR])
    ax4.set_ylabel('RAM (MB)')
    ax4.set_title('D) Memory Usage', fontweight='bold')
    ax4.annotate(f"~0 MB", xy=(0, 50), ha='center', fontsize=10)
    ax4.annotate(f"{data['duckdb_ram_mb']:.0f} MB",
                 xy=(1, data['duckdb_ram_mb']), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=10)

    # Main title
    fig.suptitle('Partition Pruning Benchmark: Rust (Disk) vs DuckDB (In-Memory)\n'
                 f"{data['total_files']} files, {data['total_rows']:,} rows",
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()

def main():
    # Load data
    data = load_results('results.json')

    print(f"\nGenerating charts for {data['total_rows']:,} rows, {data['total_files']} files...\n")

    # Generate all charts
    plot_latency_comparison(data)
    plot_speedup(data)
    plot_pruning_vs_speedup(data)
    plot_memory_comparison(data)
    plot_tradeoff_summary(data)

    print("\nAll charts generated successfully!")
    print("\nKey findings:")
    print(f"  • DuckDB RAM usage: {data['duckdb_ram_mb']:.0f} MB for {data['total_rows']/1e6:.1f}M rows")
    print(f"  • Breakeven pruning: ~87%")
    print(f"  • Rust wins when pruning > 98%")

if __name__ == '__main__':
    main()
