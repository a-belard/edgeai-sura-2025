"""Aggregate benchmark result JSON/CSV files into a single Excel workbook.

Usage (from repo root, after activating venv):
    python inference_evaluation/aggregate_results_to_excel.py \
        --results-root results \
        --output results/EdgeAI_Benchmark_Aggregated.xlsx

What it does:
 1. Scans platform subdirectories under the results root (e.g. pc/, rpi/, googlecollab/ ...).
 2. Locates model result JSON files (SqueezeNet / MobileNetV2) with flexible patterns:
       * *squeezenet*results*.json, *mobilenetv2*results*.json
       * Fallback generic *squeezenet*.json (handles squuezenet.json typo)
 3. Extracts key KPI fields (latency stats, memory, CPU, cold/warm, throughput, efficiency ratios).
 4. Builds two summary tables:
       a) all_runs: one row per JSON artifact
       b) latest_per_platform: most recent per (Platform, Model)
 5. Optionally merges any existing *_summary_*.csv files.
 6. Writes an Excel workbook with separate sheets:
       - Summary_All_Runs
       - Summary_Latest
       - CSV_Summaries (concatenated existing summary CSVs if found)
       - Latency_Distributions (long-form per inference timing, truncated via --max-latency-rows)

Dependencies: pandas, openpyxl (or xlsxwriter). If openpyxl missing, instruct user.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ----------------------------- Data Structures ----------------------------- #

@dataclass
class ResultRecord:
    platform: str
    model: str
    file: Path
    timestamp: datetime
    mean_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    std_latency_ms: Optional[float]
    min_latency_ms: Optional[float]
    max_latency_ms: Optional[float]
    mean_memory_mb: Optional[float]
    peak_memory_mb: Optional[float]
    mean_cpu_percent: Optional[float]
    peak_cpu_percent: Optional[float]
    model_size_mb: Optional[float]
    quantized: Optional[bool]
    device_type: Optional[str]
    cold_start_overhead_ms: Optional[float]
    cold_start_mean_ms: Optional[float]
    warm_start_mean_ms: Optional[float]
    max_throughput_img_s: Optional[float]
    efficiency_0_ms: Optional[float]
    efficiency_500_ms: Optional[float]
    efficiency_1000_ms: Optional[float]


# ----------------------------- Helper Functions ---------------------------- #

TIMESTAMP_FILENAME_RE = re.compile(r"(\d{8}_\d{6})")


def parse_timestamp(json_data: Dict[str, Any], file_path: Path) -> datetime:
    # 1. Try JSON timestamp
    ts = json_data.get("timestamp")
    if ts:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            pass
    # 2. Try filename pattern
    m = TIMESTAMP_FILENAME_RE.search(file_path.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    # 3. Fallback to file modified time
    return datetime.fromtimestamp(file_path.stat().st_mtime)


def safe_get(d: Dict[str, Any], *keys, default=None):  # type: ignore[override]
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_record(platform: str, file_path: Path) -> Optional[ResultRecord]:
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {file_path}: {e}")
        return None

    stats = safe_get(data, "core_performance", "statistics", default={}) or {}
    batch = safe_get(data, "batch_processing", default={}) or {}
    latency_analysis = safe_get(data, "latency_analysis", default={}) or {}
    cold_warm_stats = safe_get(data, "cold_warm_analysis", "statistics", default={}) or {}
    model_info = safe_get(data, "model_info", default={}) or {}

    # Max throughput across batch sizes
    max_throughput = None
    if isinstance(batch, dict):
        for entry in batch.values():
            if isinstance(entry, dict) and "throughput" in entry:
                v = entry.get("throughput")
                if isinstance(v, (int, float)):
                    max_throughput = v if max_throughput is None else max(max_throughput, v)

    # Efficiency ratios
    eff_0 = safe_get(latency_analysis, "0", "efficiency_ratio")
    eff_500 = safe_get(latency_analysis, "500", "efficiency_ratio")
    eff_1000 = safe_get(latency_analysis, "1000", "efficiency_ratio")

    model_name = model_info.get("name") or infer_model_from_filename(file_path.name)

    return ResultRecord(
        platform=platform,
        model=model_name,
        file=file_path,
        timestamp=parse_timestamp(data, file_path),
        mean_latency_ms=stats.get("mean_inference_time"),
        p95_latency_ms=stats.get("p95_inference_time"),
        p99_latency_ms=stats.get("p99_inference_time"),
        std_latency_ms=stats.get("std_inference_time"),
        min_latency_ms=stats.get("min_inference_time"),
        max_latency_ms=stats.get("max_inference_time"),
        mean_memory_mb=stats.get("mean_memory_usage"),
        peak_memory_mb=stats.get("peak_memory_usage"),
        mean_cpu_percent=stats.get("mean_cpu_usage"),
        peak_cpu_percent=stats.get("peak_cpu_usage"),
        model_size_mb=model_info.get("size_mb"),
        quantized=model_info.get("quantized"),
        device_type=model_info.get("device"),
        cold_start_overhead_ms=cold_warm_stats.get("startup_overhead"),
        cold_start_mean_ms=cold_warm_stats.get("cold_start_mean"),
        warm_start_mean_ms=cold_warm_stats.get("warm_start_mean"),
        max_throughput_img_s=max_throughput,
        efficiency_0_ms=eff_0,
        efficiency_500_ms=eff_500,
        efficiency_1000_ms=eff_1000,
    )


def infer_model_from_filename(name: str) -> str:
    n = name.lower()
    if "mobilenet" in n:
        return "MobileNetV2"
    if "squeezenet" in n or "squuezenet" in n:
        return "SqueezeNet"
    return "Unknown"


def discover_json_files(results_root: Path) -> List[Path]:
    patterns = [
        "**/*squeezenet*results*.json",
        "**/*mobilenetv2*results*.json",
        "**/*squeezenet*.json",       # generic SqueezeNet fallback
        "**/*squuezenet*.json",       # typo variant fallback
        "**/mobilenetv2.json",        # generic MobileNetV2 fallback (no timestamp)
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(results_root.glob(pat))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for f in sorted(files):
        if f not in seen:
            unique.append(f)
            seen.add(f)
    if not unique:
        print("[INFO] No JSON files matched any pattern.")
    else:
        print(f"[INFO] Matched {len(unique)} JSON files after deduplication.")
    return unique


def build_latency_long_form(records: List[ResultRecord], max_rows: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    count = 0
    for rec in records:
        try:
            with open(rec.file) as f:
                data = json.load(f)
        except Exception:
            continue
        times = safe_get(data, "core_performance", "inference_times", default=[])
        if not isinstance(times, list):
            continue
        for idx, t in enumerate(times):
            rows.append({
                "Platform": rec.platform,
                "Model": rec.model,
                "Timestamp": rec.timestamp,
                "RunIndex": idx,
                "Latency_ms": t,
            })
            count += 1
            if max_rows and count >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def load_summary_csvs(results_root: Path) -> pd.DataFrame:
    csvs = list(results_root.glob("**/*_summary_*.csv"))
    frames = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
            df.insert(0, "_source_file", str(c.relative_to(results_root)))
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading summary CSV {c}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def records_to_dataframe(records: List[ResultRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records])


# ----------------------------- Main Routine -------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results into Excel workbook")
    parser.add_argument("--results-root", default="results", help="Root directory containing platform subfolders")
    parser.add_argument("--output", default="results/EdgeAI_Benchmark_Aggregated.xlsx", help="Output Excel file path")
    parser.add_argument("--max-latency-rows", type=int, default=5000, help="Cap rows in latency distribution sheet (0 = no cap)")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")

    json_files = discover_json_files(results_root)
    if not json_files:
        raise SystemExit("No JSON result files discovered.")

    print(f"Discovered {len(json_files)} JSON files")

    # Derive platform from immediate parent directory name (pc, rpi, googlecollab, etc.)
    records: List[ResultRecord] = []
    for jf in json_files:
        platform = jf.parent.name
        rec = extract_record(platform, jf)
        if rec:
            records.append(rec)

    if not records:
        raise SystemExit("No valid result records extracted.")

    all_df = records_to_dataframe(records)
    all_df.sort_values(["model", "platform", "timestamp"], inplace=True)

    # Latest per (platform, model)
    latest_df = (all_df.sort_values("timestamp")
                      .groupby(["platform", "model"], as_index=False)
                      .tail(1)
                      .reset_index(drop=True))

    # Summary CSVs (if any)
    csv_df = load_summary_csvs(results_root)

    # Latency long form (optional)
    latency_df = build_latency_long_form(records, args.max_latency_rows)

    # Write Excel
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build indicator-specific pivot sheets (latest snapshot focus)
    def _prep(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        keep = [c for c in cols if c in df.columns]
        return df[keep].copy()

    latest_base = latest_df.copy()
    # Standardize column casing for clarity
    latest_base.rename(columns={
        'platform':'Platform','model':'Model','timestamp':'Timestamp'
    }, inplace=True)

    indicator_configs = [
        ("Latency_Stats", ["Platform","Model","Timestamp","mean_latency_ms","p95_latency_ms","p99_latency_ms","std_latency_ms","min_latency_ms","max_latency_ms"]),
        ("Memory_Stats", ["Platform","Model","Timestamp","mean_memory_mb","peak_memory_mb"]),
        ("CPU_Stats", ["Platform","Model","Timestamp","mean_cpu_percent","peak_cpu_percent"]),
        ("Throughput", ["Platform","Model","Timestamp","max_throughput_img_s"]),
        ("Efficiency", ["Platform","Model","Timestamp","efficiency_0_ms","efficiency_500_ms","efficiency_1000_ms"]),
        ("ColdWarm", ["Platform","Model","Timestamp","cold_start_mean_ms","warm_start_mean_ms","cold_start_overhead_ms"]),
        ("Model_Info", ["Platform","Model","Timestamp","model_size_mb","quantized","device_type"]),
    ]

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="Summary_All_Runs", index=False)
            latest_df.to_excel(writer, sheet_name="Summary_Latest", index=False)
            # Indicator sheets
            for sheet_name, cols in indicator_configs:
                df_ind = _prep(latest_base, cols)
                if not df_ind.empty:
                    df_ind.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            if not csv_df.empty:
                csv_df.to_excel(writer, sheet_name="CSV_Summaries", index=False)
            if not latency_df.empty:
                latency_df.to_excel(writer, sheet_name="Latency_Distributions", index=False)
    except ModuleNotFoundError:
        raise SystemExit("openpyxl is required to write Excel. Install with: pip install openpyxl")

    print(f"Workbook written: {output_path}")
    print("Sheets:")
    print(" - Summary_All_Runs")
    print(" - Summary_Latest")
    if not csv_df.empty:
        print(" - CSV_Summaries")
    if not latency_df.empty:
        print(" - Latency_Distributions")


if __name__ == "__main__":
    main()
