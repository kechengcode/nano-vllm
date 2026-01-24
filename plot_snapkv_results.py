import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "snapkv_sweep_results.csv"
OUTPUT_IMAGE = "snapkv_throughput_chart.png"

def plot_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Separate SnapKV=ON data
    snapkv_on = df[df["snapkv"] == "ON"].copy()
    
    # Ensure data types
    snapkv_on["compression_rate"] = snapkv_on["compression_rate"].astype(int)
    snapkv_on["throughput"] = snapkv_on["throughput"].astype(float)
    snapkv_on["input_len"] = snapkv_on["input_len"].astype(int)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    
    # Get unique input lengths and sort them
    input_lengths = sorted(snapkv_on["input_len"].unique())
    
    # Color map
    colors = plt.cm.viridis(range(0, 256, int(256/len(input_lengths))))
    
    for i, length in enumerate(input_lengths):
        subset = snapkv_on[snapkv_on["input_len"] == length].sort_values("compression_rate")
        plt.plot(
            subset["compression_rate"], 
            subset["throughput"], 
            marker='o', 
            label=f"Input Length {length}",
            color=colors[i]
        )
        
        # Add baseline (SnapKV=OFF) as a horizontal dashed line or single point?
        # User requested separate testing for SnapKV OFF.
        # Let's see if we have baseline data for this length.
        baseline = df[(df["snapkv"] == "OFF") & (df["input_len"] == length)]
        if not baseline.empty:
            base_tpt = baseline.iloc[0]["throughput"]
            plt.axhline(y=base_tpt, color=colors[i], linestyle='--', alpha=0.5, label=f"Baseline (Input {length})")

    plt.xlabel("Compression Multiplier (x)")
    plt.ylabel("Total Throughput (tok/s)")
    plt.title("SnapKV Throughput vs. Compression Rate for Varying Input Lengths")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(sorted(snapkv_on["compression_rate"].unique()))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Chart saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_results()
