"""
analyze_telemetry.py
====================================================
Reads telemetry.db (produced by gesture_hud.py) and generates the
three experiment outputs for the Case Study Report:

  1. Latency Profiling Test      -> latency_profile.png + printed stats
  2. Environmental Lighting Test -> lighting_benchmark.png (needs benchmark_summary rows,
                                     recorded in-app by pressing 'b' to start/stop a run
                                     under each lighting condition)
  3. False Positive (Midas Touch) Test -> printed false-positive rate per lighting run

Run AFTER you've collected some data with gesture_hud.py:
    python analyze_telemetry.py
Outputs are written next to this script.
"""

import sqlite3
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = "telemetry.db"
PERCEPTUAL_LAG_LIMIT_MS = 60


def load_tables(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    logs = pd.read_sql_query("SELECT * FROM telemetry_logs", conn)
    summary = pd.read_sql_query("SELECT * FROM benchmark_summary", conn)
    conn.close()
    return logs, summary


def latency_profiling_test(logs):
    if logs.empty:
        print("[latency test] No rows in telemetry_logs yet — run gesture_hud.py first.")
        return

    window = logs.tail(500).copy()
    window["frame_idx"] = range(len(window))

    stats = window["total_latency_ms"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    pct_under_limit = (window["total_latency_ms"] < PERCEPTUAL_LAG_LIMIT_MS).mean() * 100

    print("\n=== Experiment 1: Latency Profiling Test ===")
    print(f"Frames analyzed: {len(window)}")
    print(f"Mean total latency:   {stats['mean']:.2f} ms")
    print(f"Median (p50):         {stats['50%']:.2f} ms")
    print(f"p90:                  {stats['90%']:.2f} ms")
    print(f"p95:                  {stats['95%']:.2f} ms")
    print(f"p99:                  {stats['99%']:.2f} ms")
    print(f"Max:                  {stats['max']:.2f} ms")
    print(f"% of frames under {PERCEPTUAL_LAG_LIMIT_MS}ms limit: {pct_under_limit:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(window["frame_idx"], window["capture_latency_ms"], label="Capture", linewidth=1)
    ax.plot(window["frame_idx"], window["inference_latency_ms"], label="Inference", linewidth=1)
    ax.plot(window["frame_idx"], window["total_latency_ms"], label="Total", linewidth=1.5, color="black")
    ax.axhline(PERCEPTUAL_LAG_LIMIT_MS, color="red", linestyle="--", linewidth=1,
               label=f"{PERCEPTUAL_LAG_LIMIT_MS}ms perceptual limit")
    ax.set_xlabel("Frame index (most recent 500)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Profiling Test — Capture / Inference / Total")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("latency_profile.png", dpi=150)
    print("Saved chart -> latency_profile.png")


def lighting_benchmark_test(summary):
    if summary.empty:
        print("\n[lighting test] No rows in benchmark_summary yet.")
        print("In gesture_hud.py: press 'l' to select a lighting condition, "
              "'b' to start a benchmark run, wait, then 'b' again to stop and save it.")
        return

    print("\n=== Experiment 2: Environmental Lighting Benchmark ===")
    grouped = summary.groupby("lighting_condition").agg(
        runs=("test_id", "count"),
        avg_latency_ms=("avg_latency_ms", "mean"),
        total_frames=("total_frames_tested", "sum"),
        total_false_positives=("false_positives", "sum"),
    ).reset_index()
    grouped["false_positive_rate_pct"] = (
        grouped["total_false_positives"] / grouped["total_frames"].replace(0, pd.NA) * 100
    ).fillna(0)
    print(grouped.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(grouped["lighting_condition"], grouped["avg_latency_ms"], color="#2b6cb0")
    ax.set_ylabel("Avg total latency (ms)")
    ax.set_title("Latency by Lighting Condition")
    fig.tight_layout()
    fig.savefig("lighting_benchmark.png", dpi=150)
    print("Saved chart -> lighting_benchmark.png")


def false_positive_test(summary):
    if summary.empty:
        return
    print("\n=== Experiment 3: False Positive (Midas Touch) Test ===")
    for _, row in summary.iterrows():
        rate = (row["false_positives"] / row["total_frames_tested"] * 100) if row["total_frames_tested"] else 0
        print(f"[{row['lighting_condition']}] run {row['test_id']}: "
              f"{row['false_positives']} false positives / {row['total_frames_tested']} frames "
              f"({rate:.2f}%)")


def main():
    logs, summary = load_tables()
    latency_profiling_test(logs)
    lighting_benchmark_test(summary)
    false_positive_test(summary)


if __name__ == "__main__":
    main()
