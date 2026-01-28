import os
import math
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "/root/nano-vllm/snapkv_throughput_and_maxbatch_v3.csv"
OUTPUT_IMAGE = "qwen3-8b-snapkv_throughput_chart_v3.png"

def plot_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty.")
        return

    # Normalize columns
    df = df.copy()
    if "snapkv" not in df.columns or "input_len" not in df.columns or "compression_rate" not in df.columns or "throughput" not in df.columns:
        print("CSV must contain columns: snapkv, input_len, compression_rate, throughput")
        return

    df["snapkv"] = df["snapkv"].astype(str)
    df["input_len"] = pd.to_numeric(df["input_len"], errors="coerce").astype(pd.Int64Dtype())
    df["compression_rate"] = pd.to_numeric(df["compression_rate"], errors="coerce")
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")

    df = df.dropna(subset=["input_len", "compression_rate", "throughput"])
    if df.empty:
        print("No valid numeric data after cleaning.")
        return

    # Exclude input length 10000 as requested
    #df = df[df["input_len"] != 10000]
    input_lengths = sorted(df["input_len"].unique())
    n = len(input_lengths)
    if n == 0:
        print("No input lengths found.")
        return

    # Fixed layout: 2x2 as requested
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, length in enumerate(input_lengths):
        ax = axes_flat[idx]
        subset_on = df[(df["snapkv"].str.upper() == "ON") & (df["input_len"] == length)].copy()
        if subset_on.empty:
            ax.text(0.5, 0.5, "No ON data", ha="center", va="center")
            ax.set_title(f"Input Length {length}")
            ax.set_xlabel("Compression Multiplier (x)")
            ax.set_ylabel("Total Throughput (tok/s)")
            continue

        subset_on = subset_on.sort_values("compression_rate")
        ax.plot(subset_on["compression_rate"], subset_on["throughput"], marker='o', label="SparseKV ON")

        # Baseline OFF for this input length (take mean if multiple)
        baseline = df[(df["snapkv"].str.upper() == "OFF") & (df["input_len"] == length)]
        if not baseline.empty:
            base_tpt = baseline["throughput"].mean()
            ax.axhline(y=base_tpt, color="gray", linestyle='--', alpha=0.7, label="Baseline (OFF)")

        ax.set_title(f"Input Length {length}")
        ax.set_xlabel("Compression Multiplier (x)")
        ax.set_ylabel("Total Throughput (tok/s)")
        ax.grid(True, ls='-', alpha=0.2)
        xticks = sorted(subset_on["compression_rate"].unique())
        if xticks:
            ax.set_xticks(xticks)
        ax.legend()

    # Hide unused axes
    for j in range(n, rows * cols):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Chart saved to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    plot_results()
