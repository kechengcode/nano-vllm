import os
import math
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "/root/nano-vllm/snapkv_throughput_and_maxbatch_v3.csv"
OUTPUT_IMAGE = "snapkv_maxbatch_chart.png"

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
    # We need: snapkv, input_len, compression_rate, max_batch_size
    required_cols = ["snapkv", "input_len", "compression_rate", "max_batch_size"]
    for col in required_cols:
        if col not in df.columns:
            print(f"CSV must contain column: {col}")
            return

    df["snapkv"] = df["snapkv"].astype(str)
    df["input_len"] = pd.to_numeric(df["input_len"], errors="coerce").astype(pd.Int64Dtype())
    df["compression_rate"] = pd.to_numeric(df["compression_rate"], errors="coerce")
    df["max_batch_size"] = pd.to_numeric(df["max_batch_size"], errors="coerce")

    # Drop rows where critical data is NaN
    df = df.dropna(subset=["input_len", "compression_rate", "max_batch_size"])
    
    if df.empty:
        print("No valid numeric data after cleaning.")
        return

    input_lengths = sorted(df["input_len"].unique())
    n = len(input_lengths)
    if n == 0:
        print("No input lengths found.")
        return

    # Layout: similar to v2, use 3 columns
    cols = 3
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, length in enumerate(input_lengths):
        ax = axes_flat[idx]
        
        # 1. Plot SnapKV ON data
        subset_on = df[(df["snapkv"].str.upper() == "ON") & (df["input_len"] == length)].copy()
        
        if not subset_on.empty:
            subset_on = subset_on.sort_values("compression_rate")
            ax.plot(subset_on["compression_rate"], subset_on["max_batch_size"], 
                    marker='o', label="SparseKV ON", color='tab:blue', linewidth=2)
            
            # Set X ticks explicitely to the compression rates present
            xticks = sorted(subset_on["compression_rate"].unique())
            ax.set_xticks(xticks)

        # 2. Plot Baseline OFF data (as horizontal line)
        baseline = df[(df["snapkv"].str.upper() == "OFF") & (df["input_len"] == length)]
        if not baseline.empty:
            base_bs = baseline["max_batch_size"].mean()
            ax.axhline(y=base_bs, color="tab:orange", linestyle='--', 
                       linewidth=2, label=f"Baseline (OFF): {int(base_bs)}")

        ax.set_title(f"Input Length {length}")
        ax.set_xlabel("Compression Rate (x)")
        ax.set_ylabel("Max Batch Size")
        ax.grid(True, ls='-', alpha=0.2)
        ax.legend()

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Chart saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_results()
