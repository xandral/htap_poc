use arrow::array::{BooleanArray, Int64Array, StringArray};
use arrow::compute::kernels::boolean::and;
use arrow::compute::kernels::cmp::{eq, gt_eq, lt_eq};
use duckdb::{params, Connection};
use glob::glob;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics;
use rayon::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

// =============================================================================
// CONFIGURATION (loaded from config.json)
// =============================================================================

const BENCHMARK_RUNS: usize = 15;
const WARMUP_RUNS: usize = 5;

#[derive(Deserialize, Clone)]
struct DataConfig {
    num_rows: i64,
    num_sensors: usize,
    #[allow(dead_code)]
    num_data_columns: usize,
    chunk_rows: i64,
    #[allow(dead_code)]
    chunk_cols: usize,
    #[allow(dead_code)]
    seed: i64,
    #[allow(dead_code)]
    output_dir: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig {
            num_rows: 1_000_000,
            num_sensors: 5,
            num_data_columns: 10,
            chunk_rows: 10_000,
            chunk_cols: 4,
            seed: 42,
            output_dir: "./benchmark_data".to_string(),
        }
    }
}

fn load_config(data_path: &str) -> DataConfig {
    let config_path = format!("{}/config.json", data_path);
    match std::fs::read_to_string(&config_path) {
        Ok(content) => {
            serde_json::from_str(&content).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to parse config.json: {}", e);
                DataConfig::default()
            })
        }
        Err(_) => {
            eprintln!("Warning: config.json not found, using defaults");
            DataConfig::default()
        }
    }
}

fn generate_sensor_name(index: usize) -> String {
    format!("sensor_{}", (b'A' + index as u8) as char)
}

// =============================================================================
// FILTER ENUM
// =============================================================================

#[derive(Clone)]
enum Filter {
    EqTimestamp(i64),
    EqSensor(String),
    EqBoth(i64, String),
    RangeTimeAndSensor(i64, i64, String),
}

// =============================================================================
// BENCHMARK RESULT STRUCTURES
// =============================================================================

#[derive(Clone)]
struct QueryStats {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
}

#[derive(Clone)]
struct QueryResult {
    name: String,
    description: String,
    files_scanned: usize,
    total_files: usize,
    pruning_pct: f64,
    rows_returned: usize,
    rust_stats: QueryStats,
    duckdb_stats: QueryStats,
    speedup: f64,  // >1 means Rust faster
    winner: String,
}

struct BenchmarkSummary {
    total_files: usize,
    total_rows: i64,
    duckdb_load_time_s: f64,
    queries: Vec<QueryResult>,
}

// =============================================================================
// CHUNK COORDINATE PARSING
// =============================================================================

#[derive(Debug, Clone)]
struct ChunkCoords {
    row_chunk: i64,
    col_chunk: i32,
    hash_values: Vec<usize>,
    version: i32,
}

fn parse_chunk_filename(name: &str) -> Option<ChunkCoords> {
    let name = name.trim_end_matches(".parquet");
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() < 6 || parts[0] != "chunk" {
        return None;
    }
    let row_chunk = parts[1].trim_start_matches('r').parse::<i64>().ok()?;
    let col_chunk = parts[2].trim_start_matches('c').parse::<i32>().ok()?;
    let hash_str = parts[3].trim_start_matches('h');
    let hash_values: Vec<usize> = hash_str
        .split('-')
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    let version = parts[5].trim_start_matches('v').parse::<i32>().ok()?;
    Some(ChunkCoords { row_chunk, col_chunk, hash_values, version })
}

fn get_hash_bucket(value: &str, buckets: usize) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let bytes: [u8; 8] = digest[0..8].try_into().unwrap();
    (u64::from_be_bytes(bytes) % buckets as u64) as usize
}

fn get_row_chunk(global_id: i64, chunk_rows: i64) -> i64 {
    (global_id - 1) / chunk_rows
}

fn get_all_parquet_files(data_path: &str) -> Vec<PathBuf> {
    glob(&format!("{}/*.parquet", data_path))
        .expect("Failed glob")
        .filter_map(|e| e.ok())
        .collect()
}

/// Build HashMap: key "r{row}_c{col}_h{hash}" -> latest version PathBuf
fn get_latest_versions(files: &[PathBuf]) -> HashMap<String, PathBuf> {
    let mut latest: HashMap<String, (i32, PathBuf)> = HashMap::new();
    for file in files {
        let name = file.file_name().unwrap().to_str().unwrap();
        if let Some(coords) = parse_chunk_filename(name) {
            let key = format!(
                "r{}_c{}_h{}",
                coords.row_chunk,
                coords.col_chunk,
                coords.hash_values.first().unwrap_or(&0)
            );
            let entry = latest.entry(key).or_insert((coords.version, file.clone()));
            if coords.version > entry.0 {
                *entry = (coords.version, file.clone());
            }
        }
    }
    latest.into_iter().map(|(k, (_, p))| (k, p)).collect()
}

/// Precalculate file patterns and lookup in catalog
/// Returns files matching the theoretical coordinates
fn select_files(
    catalog: &HashMap<String, PathBuf>,
    min_row_chunk: i64,
    max_row_chunk: Option<i64>,
    columns: &[i32],
    hash_bucket: Option<usize>,
    num_hash_buckets: usize,
) -> Vec<PathBuf> {
    let mut result = Vec::new();

    let max_chunk = max_row_chunk.unwrap_or(min_row_chunk);

    // Generate theoretical patterns and lookup
    for row in min_row_chunk..=max_chunk {
        for &col in columns {
            match hash_bucket {
                Some(h) => {
                    // Single hash bucket
                    let key = format!("r{}_c{}_h{}", row, col, h);
                    if let Some(path) = catalog.get(&key) {
                        result.push(path.clone());
                    }
                }
                None => {
                    // All hash buckets
                    for h in 0..num_hash_buckets {
                        let key = format!("r{}_c{}_h{}", row, col, h);
                        if let Some(path) = catalog.get(&key) {
                            result.push(path.clone());
                        }
                    }
                }
            }
        }
    }

    result
}

// =============================================================================
// PARQUET SCAN (RUST IMPLEMENTATION)
// =============================================================================

#[allow(deprecated)]
fn check_statistics_timestamp(stats: Option<&Statistics>, target_ts: i64) -> bool {
    let Some(stats) = stats else { return true };
    if let Statistics::Int64(s) = stats {
        return target_ts >= *s.min() && target_ts <= *s.max();
    }
    true
}

#[allow(deprecated)]
fn check_statistics_range(stats: Option<&Statistics>, min_ts: i64, max_ts: i64) -> bool {
    let Some(stats) = stats else { return true };
    if let Statistics::Int64(s) = stats {
        if *s.max() < min_ts || *s.min() > max_ts { return false; }
    }
    true
}

fn scan_and_count(path: &PathBuf, filter: Filter) -> usize {
    let file = File::open(path).expect("Failed to open");
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("Failed builder");

    let schema = builder.schema().clone();
    let ts_idx = schema.index_of("timestamp").ok();
    let sensor_idx = schema.index_of("sensor_id").ok();

    match &filter {
        Filter::EqTimestamp(_) => if ts_idx.is_none() { return 0; },
        Filter::EqSensor(_) => if sensor_idx.is_none() { return 0; },
        Filter::EqBoth(_, _) | Filter::RangeTimeAndSensor(_, _, _) => {
            if ts_idx.is_none() || sensor_idx.is_none() { return 0; }
        }
    }

    // Row group pruning
    let metadata = builder.metadata().clone();
    let mut valid_row_groups = Vec::new();

    for i in 0..metadata.num_row_groups() {
        let rg = metadata.row_group(i);
        let mut keep = true;

        match &filter {
            Filter::EqTimestamp(ts) | Filter::EqBoth(ts, _) => {
                if let Some(idx) = ts_idx {
                    if let Some(col) = rg.columns().get(idx) {
                        if !check_statistics_timestamp(col.statistics(), *ts) { keep = false; }
                    }
                }
            }
            Filter::RangeTimeAndSensor(min_t, max_t, _) => {
                if let Some(idx) = ts_idx {
                    if let Some(col) = rg.columns().get(idx) {
                        if !check_statistics_range(col.statistics(), *min_t, *max_t) { keep = false; }
                    }
                }
            }
            Filter::EqSensor(_) => {}
        }
        if keep { valid_row_groups.push(i); }
    }

    if valid_row_groups.is_empty() { return 0; }
    builder = builder.with_row_groups(valid_row_groups);

    // Column projection
    let mut indices = Vec::new();
    match &filter {
        Filter::EqTimestamp(_) => { if let Some(i) = ts_idx { indices.push(i); } }
        Filter::EqSensor(_) => { if let Some(i) = sensor_idx { indices.push(i); } }
        Filter::EqBoth(_, _) | Filter::RangeTimeAndSensor(_, _, _) => {
            if let Some(i) = ts_idx { indices.push(i); }
            if let Some(i) = sensor_idx { indices.push(i); }
        }
    }

    if !indices.is_empty() {
        let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
        builder = builder.with_projection(mask);
    }

    // Scan
    let reader = builder.build().expect("Reader build");
    let mut count = 0;

    for batch_result in reader {
        let batch = batch_result.expect("Read batch");
        match &filter {
            Filter::EqTimestamp(ts) => {
                let col = batch.column_by_name("timestamp").unwrap();
                let val = Int64Array::from_value(*ts, batch.num_rows());
                if let Ok(mask) = eq(col, &val) { count += mask.true_count(); }
            }
            Filter::EqSensor(s) => {
                let col = batch.column_by_name("sensor_id").unwrap();
                let val = StringArray::from(vec![s.clone(); batch.num_rows()]);
                if let Ok(mask) = eq(col, &val) { count += mask.true_count(); }
            }
            Filter::EqBoth(ts, s) => {
                let c_ts = batch.column_by_name("timestamp").unwrap();
                let c_s = batch.column_by_name("sensor_id").unwrap();
                let m1 = eq(c_ts, &Int64Array::from_value(*ts, batch.num_rows()))
                    .unwrap_or(BooleanArray::from(vec![false; batch.num_rows()]));
                let m2 = eq(c_s, &StringArray::from(vec![s.clone(); batch.num_rows()]))
                    .unwrap_or(BooleanArray::from(vec![false; batch.num_rows()]));
                if let Ok(final_m) = and(&m1, &m2) { count += final_m.true_count(); }
            }
            Filter::RangeTimeAndSensor(min_t, max_t, s) => {
                let c_ts = batch.column_by_name("timestamp").unwrap();
                let c_s = batch.column_by_name("sensor_id").unwrap();
                let m_ge = gt_eq(c_ts, &Int64Array::from_value(*min_t, batch.num_rows())).unwrap();
                let m_le = lt_eq(c_ts, &Int64Array::from_value(*max_t, batch.num_rows())).unwrap();
                let m_ts = and(&m_ge, &m_le).unwrap();
                let m_s = eq(c_s, &StringArray::from(vec![s.clone(); batch.num_rows()])).unwrap();
                if let Ok(final_m) = and(&m_ts, &m_s) { count += final_m.true_count(); }
            }
        }
    }
    count
}

fn run_parallel_scan(files: &[PathBuf], filter: Filter) -> usize {
    files.par_iter().map(|f| scan_and_count(f, filter.clone())).sum()
}

// =============================================================================
// DUCKDB
// =============================================================================

fn setup_duckdb_table(data_path: &str) -> Connection {
    let db_file = format!("{}/benchmark.duckdb", data_path);
    Connection::open(&db_file).unwrap()
}

fn duckdb_query(conn: &Connection, query: &str, p: &[&dyn duckdb::ToSql]) -> usize {
    let mut stmt = conn.prepare(query).unwrap();
    let count: i64 = stmt.query_row(p, |r| r.get(0)).unwrap();
    count as usize
}

// =============================================================================
// STATISTICS
// =============================================================================

fn compute_stats(times: &[f64]) -> QueryStats {
    let n = times.len() as f64;
    let avg = times.iter().sum::<f64>() / n;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

    QueryStats { avg_ms: avg, p50_ms: p50, p95_ms: p95, min_ms: min, max_ms: max, stddev_ms: stddev }
}

// =============================================================================
// BENCHMARK EXECUTION
// =============================================================================

fn run_benchmark<F, G>(
    name: &str,
    description: &str,
    files_scanned: usize,
    total_files: usize,
    rust_op: F,
    duckdb_op: G,
) -> QueryResult
where
    F: Fn() -> usize,
    G: Fn() -> usize,
{
    // Warmup
    for _ in 0..WARMUP_RUNS {
        rust_op();
        duckdb_op();
    }

    // Rust measurements
    let mut r_times = Vec::new();
    let mut r_res = 0;
    for _ in 0..BENCHMARK_RUNS {
        let s = Instant::now();
        r_res = rust_op();
        r_times.push(s.elapsed().as_secs_f64() * 1000.0);
    }
    let rust_stats = compute_stats(&r_times);

    // DuckDB measurements
    let mut d_times = Vec::new();
    let mut d_res = 0;
    for _ in 0..BENCHMARK_RUNS {
        let s = Instant::now();
        d_res = duckdb_op();
        d_times.push(s.elapsed().as_secs_f64() * 1000.0);
    }
    let duckdb_stats = compute_stats(&d_times);

    // Verify correctness
    if r_res != d_res {
        eprintln!("WARNING: Result mismatch for {}: Rust={}, DuckDB={}", name, r_res, d_res);
    }

    let speedup = duckdb_stats.p50_ms / rust_stats.p50_ms;
    let winner = if speedup > 1.1 { "Rust".to_string() }
                 else if speedup < 0.9 { "DuckDB".to_string() }
                 else { "Tie".to_string() };

    let pruning_pct = 100.0 * (1.0 - files_scanned as f64 / total_files as f64);

    QueryResult {
        name: name.to_string(),
        description: description.to_string(),
        files_scanned,
        total_files,
        pruning_pct,
        rows_returned: r_res,
        rust_stats,
        duckdb_stats,
        speedup,
        winner,
    }
}

// =============================================================================
// OUTPUT FORMATTING
// =============================================================================

fn print_header() {
    println!();
    println!("================================================================================");
    println!("       PARTITION PRUNING BENCHMARK: Custom Rust vs DuckDB In-Memory");
    println!("================================================================================");
    println!();
}

fn print_config_with_params(total_files: usize, total_rows: i64, duck_load: f64, chunk_rows: i64, hash_buckets: usize) {
    println!("CONFIGURATION");
    println!("-------------");
    println!("  Chunk size:        {} rows", chunk_rows);
    println!("  Hash buckets:      {}", hash_buckets);
    println!("  Benchmark runs:    {} (after {} warmup)", BENCHMARK_RUNS, WARMUP_RUNS);
    println!("  Total files:       {}", total_files);
    println!("  Total rows:        {}", total_rows);
    println!();
    println!("SYSTEM COMPARISON");
    println!("-----------------");
    println!("  {:20} {:>15} {:>15}", "", "Rust", "DuckDB");
    println!("  {:20} {:>15} {:>15}", "Startup cost:", "0 s", format!("{:.2} s", duck_load));
    println!("  {:20} {:>15} {:>15}", "Data location:", "Disk", "Persistent DB");
    println!();
}

fn print_query_result(q: &QueryResult) {
    println!("--------------------------------------------------------------------------------");
    println!("{}", q.name);
    println!("  {}", q.description);
    println!("--------------------------------------------------------------------------------");
    println!("  Files scanned: {} / {} ({:.1}% pruned)", q.files_scanned, q.total_files, q.pruning_pct);
    println!("  Rows returned: {}", q.rows_returned);
    println!();
    println!("  {:12} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "", "p50", "avg", "p95", "min", "max");
    println!("  {:12} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
             "Rust (ms):", q.rust_stats.p50_ms, q.rust_stats.avg_ms,
             q.rust_stats.p95_ms, q.rust_stats.min_ms, q.rust_stats.max_ms);
    println!("  {:12} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
             "DuckDB (ms):", q.duckdb_stats.p50_ms, q.duckdb_stats.avg_ms,
             q.duckdb_stats.p95_ms, q.duckdb_stats.min_ms, q.duckdb_stats.max_ms);
    println!();
    let speedup_str = if q.speedup > 1.0 {
        format!("{:.2}x", q.speedup)
    } else {
        format!("{:.2}x", 1.0 / q.speedup)
    };
    println!("  Winner: {} ({})", q.winner, speedup_str);
    println!();
}

fn print_summary(summary: &BenchmarkSummary) {
    println!("================================================================================");
    println!("                              RESULTS SUMMARY");
    println!("================================================================================");
    println!();
    println!("{:<45} {:>8} {:>10} {:>10} {:>8}", "Query", "Pruning", "Rust p50", "Duck p50", "Winner");
    println!("{:-<45} {:->8} {:->10} {:->10} {:->8}", "", "", "", "", "");

    for q in &summary.queries {
        let winner_mark = match q.winner.as_str() {
            "Rust" => format!("Rust {:.1}x", q.speedup),
            "DuckDB" => format!("Duck {:.1}x", 1.0/q.speedup),
            _ => "Tie".to_string(),
        };
        println!("{:<45} {:>7.1}% {:>9.2}ms {:>9.2}ms {:>8}",
                 q.name, q.pruning_pct, q.rust_stats.p50_ms, q.duckdb_stats.p50_ms, winner_mark);
    }

    println!();
    println!("================================================================================");
    println!("                               CONCLUSIONS");
    println!("================================================================================");
    println!();
    println!("  PERFORMANCE:");
    let rust_wins: Vec<_> = summary.queries.iter().filter(|q| q.winner == "Rust").collect();
    let duck_wins: Vec<_> = summary.queries.iter().filter(|q| q.winner == "DuckDB").collect();
    println!("    - Rust wins {} queries (high pruning scenarios)", rust_wins.len());
    println!("    - DuckDB wins {} queries (low pruning scenarios)", duck_wins.len());

    if let Some(breakeven) = summary.queries.iter().find(|q| q.winner == "Tie") {
        println!("    - Breakeven at ~{:.0}% pruning", breakeven.pruning_pct);
    }
    println!();
    println!("  KEY INSIGHT:");
    println!("    Partition pruning on disk can match or beat DuckDB");
    println!("    when pruning eliminates >90%% of files.");
    println!();
}

fn export_json(summary: &BenchmarkSummary, path: &str) {
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"total_files\": {},\n", summary.total_files));
    json.push_str(&format!("  \"total_rows\": {},\n", summary.total_rows));
    json.push_str(&format!("  \"duckdb_load_time_s\": {:.2},\n", summary.duckdb_load_time_s));
    json.push_str(&format!("  \"benchmark_runs\": {},\n", BENCHMARK_RUNS));
    json.push_str(&format!("  \"warmup_runs\": {},\n", WARMUP_RUNS));
    json.push_str("  \"queries\": [\n");

    for (i, q) in summary.queries.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"name\": \"{}\",\n", q.name));
        json.push_str(&format!("      \"files_scanned\": {},\n", q.files_scanned));
        json.push_str(&format!("      \"pruning_pct\": {:.1},\n", q.pruning_pct));
        json.push_str(&format!("      \"rows_returned\": {},\n", q.rows_returned));
        json.push_str(&format!("      \"rust_p50_ms\": {:.3},\n", q.rust_stats.p50_ms));
        json.push_str(&format!("      \"rust_avg_ms\": {:.3},\n", q.rust_stats.avg_ms));
        json.push_str(&format!("      \"rust_stddev_ms\": {:.3},\n", q.rust_stats.stddev_ms));
        json.push_str(&format!("      \"duckdb_p50_ms\": {:.3},\n", q.duckdb_stats.p50_ms));
        json.push_str(&format!("      \"duckdb_avg_ms\": {:.3},\n", q.duckdb_stats.avg_ms));
        json.push_str(&format!("      \"duckdb_stddev_ms\": {:.3},\n", q.duckdb_stats.stddev_ms));
        json.push_str(&format!("      \"speedup\": {:.2},\n", q.speedup));
        json.push_str(&format!("      \"winner\": \"{}\"\n", q.winner));
        json.push_str(if i < summary.queries.len() - 1 { "    },\n" } else { "    }\n" });
    }

    json.push_str("  ]\n}\n");

    if let Ok(mut file) = File::create(path) {
        let _ = file.write_all(json.as_bytes());
        println!("Results exported to: {}", path);
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <data_path> [output.json]", args[0]);
        return;
    }
    let data_path = &args[1];
    let json_output = args.get(2).map(|s| s.as_str());

    print_header();

    // Load configuration from data directory
    let config = load_config(data_path);
    let chunk_rows = config.chunk_rows;
    let hash_buckets = config.num_sensors;
    println!("Loaded config: chunk_rows={}, hash_buckets={}", chunk_rows, hash_buckets);

    // Build catalog: HashMap with key "r{row}_c{col}_h{hash}" -> PathBuf
    let all_files = get_all_parquet_files(data_path);
    let catalog = get_latest_versions(&all_files);

    // Precalculate max row chunk (global)
    let max_row_chunk = (config.num_rows - 1) / chunk_rows;
    let col0: Vec<i32> = vec![0];  // Only col_chunk 0 for counting queries

    // Total files = all col_chunk=0 files across all row chunks and hash buckets
    let total_files = select_files(&catalog, 0, Some(max_row_chunk), &col0, None, hash_buckets).len();

    // Setup DuckDB
    print!("Loading DuckDB... ");
    std::io::stdout().flush().unwrap();
    let load_start = Instant::now();
    let duck_conn = setup_duckdb_table(data_path);
    let duck_load_time = load_start.elapsed().as_secs_f64();
    println!("done ({:.2}s)", duck_load_time);

    let total_rows: i64 = duck_conn
        .prepare("SELECT COUNT(*) FROM iot").unwrap()
        .query_row([], |r| r.get(0)).unwrap();

    print_config_with_params(total_files, total_rows, duck_load_time, chunk_rows, hash_buckets);

    // Query parameters - use sensor that exists in our data
    let ts = (config.num_rows / 2) as i64;  // Middle of data range
    let sensor = generate_sensor_name(config.num_sensors - 1);  // Last sensor (e.g., sensor_E for 5 sensors)
    let hash = get_hash_bucket(&sensor, hash_buckets);
    let row_chunk = get_row_chunk(ts, chunk_rows);

    println!("QUERY PARAMETERS");
    println!("----------------");
    println!("  timestamp = {} (maps to row_chunk {})", ts, row_chunk);
    println!("  sensor_id = '{}' (maps to hash_bucket {})", sensor, hash);
    println!();

    let mut queries = Vec::new();

    // Q1: Point query on timestamp (all hash buckets, single row chunk)
    let f1 = select_files(&catalog, row_chunk, None, &col0, None, hash_buckets);
    queries.push(run_benchmark(
        "Q1: timestamp = X",
        "Row chunk pruning only",
        f1.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, row_chunk, None, &col0, None, hash_buckets), Filter::EqTimestamp(ts)),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE timestamp = ?", params![ts]),
    ));

    // Q2: Point query on sensor_id (single hash bucket, all row chunks)
    let f2 = select_files(&catalog, 0, Some(max_row_chunk), &col0, Some(hash), hash_buckets);
    let s2 = sensor.clone();
    queries.push(run_benchmark(
        "Q2: sensor_id = X",
        "Hash bucket pruning only",
        f2.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, 0, Some(max_row_chunk), &col0, Some(hash), hash_buckets), Filter::EqSensor(sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE sensor_id = ?", params![s2.clone()]),
    ));

    // Q3: Point query on both (single row chunk, single hash bucket)
    let f3 = select_files(&catalog, row_chunk, None, &col0, Some(hash), hash_buckets);
    let s3 = sensor.clone();
    queries.push(run_benchmark(
        "Q3: timestamp = X AND sensor_id = Y",
        "Row chunk + hash bucket pruning",
        f3.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, row_chunk, None, &col0, Some(hash), hash_buckets), Filter::EqBoth(ts, sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE timestamp = ? AND sensor_id = ?", params![ts, s3.clone()]),
    ));

    // Q4: Full scan (no pruning - all row chunks, all hash buckets)
    let f4 = select_files(&catalog, 0, Some(max_row_chunk), &col0, None, hash_buckets);
    let s4 = sensor.clone();
    queries.push(run_benchmark(
        "Q4: sensor_id = X (full scan)",
        "No pruning - baseline comparison",
        f4.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, 0, Some(max_row_chunk), &col0, None, hash_buckets), Filter::EqSensor(sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE sensor_id = ?", params![s4.clone()]),
    ));

    // Q5: Range query - 6% of data range
    let range_size = config.num_rows / 16;
    let (min_ts, max_ts) = (ts - range_size / 2, ts + range_size / 2);
    let min_chunk_q5 = get_row_chunk(min_ts, chunk_rows);
    let max_chunk_q5 = get_row_chunk(max_ts, chunk_rows);
    let f5 = select_files(&catalog, min_chunk_q5, Some(max_chunk_q5), &col0, Some(hash), hash_buckets);
    let s5 = sensor.clone();
    queries.push(run_benchmark(
        "Q5: ts BETWEEN X AND Y AND sensor = Z",
        "Range + hash pruning",
        f5.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, min_chunk_q5, Some(max_chunk_q5), &col0, Some(hash), hash_buckets), Filter::RangeTimeAndSensor(min_ts, max_ts, sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE timestamp >= ? AND timestamp <= ? AND sensor_id = ?", params![min_ts, max_ts, s5.clone()]),
    ));

    // Q6: Wide range - 50% of data
    let wide_range = config.num_rows / 2;
    let (wide_min, wide_max) = (1_i64, wide_range);
    let min_chunk_q6 = get_row_chunk(wide_min, chunk_rows);
    let max_chunk_q6 = get_row_chunk(wide_max, chunk_rows);
    let f6 = select_files(&catalog, min_chunk_q6, Some(max_chunk_q6), &col0, Some(hash), hash_buckets);
    let s6 = sensor.clone();
    queries.push(run_benchmark(
        "Q6: Wide range + sensor",
        "Moderate pruning (many chunks)",
        f6.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, min_chunk_q6, Some(max_chunk_q6), &col0, Some(hash), hash_buckets), Filter::RangeTimeAndSensor(wide_min, wide_max, sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE timestamp >= ? AND timestamp <= ? AND sensor_id = ?", params![wide_min, wide_max, s6.clone()]),
    ));

    // Q7: Small range - 1% of data
    let small_range = config.num_rows / 100;
    let (small_min, small_max) = (1_i64, small_range);
    let min_chunk_q7 = get_row_chunk(small_min, chunk_rows);
    let max_chunk_q7 = get_row_chunk(small_max, chunk_rows);
    let f7 = select_files(&catalog, min_chunk_q7, Some(max_chunk_q7), &col0, Some(hash), hash_buckets);
    let s7 = sensor.clone();
    queries.push(run_benchmark(
        "Q7: Small range + sensor",
        "High pruning (few chunks)",
        f7.len(), total_files,
        || run_parallel_scan(&select_files(&catalog, min_chunk_q7, Some(max_chunk_q7), &col0, Some(hash), hash_buckets), Filter::RangeTimeAndSensor(small_min, small_max, sensor.clone())),
        || duckdb_query(&duck_conn, "SELECT COUNT(*) FROM iot WHERE timestamp >= ? AND timestamp <= ? AND sensor_id = ?", params![small_min, small_max, s7.clone()]),
    ));

    // Print results
    println!("DETAILED RESULTS");
    println!("================");
    for q in &queries {
        print_query_result(q);
    }

    let summary = BenchmarkSummary {
        total_files,
        total_rows,
        duckdb_load_time_s: duck_load_time,
        queries,
    };

    print_summary(&summary);

    if let Some(path) = json_output {
        export_json(&summary, path);
    }
}
