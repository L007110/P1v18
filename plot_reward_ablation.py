import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

LOG_DIR = "training_results"

# 定义文件映射
FILES = {
    "Baseline (Full Reward)": "scalability_P1v16_BASELINE.csv",
    "R1: No V2I Penalty": "scalability_P1v16_NO_V2I_PENALTY.csv",
    "R2: No Delay Reward": "scalability_P1v16_NO_DELAY_REWARD.csv",
    "R3: Only SINR (Greedy)": "scalability_P1v16_ONLY_SINR.csv"
}

PLOT_FILE = "reward_ablation_analysis.png"

def plot_analysis():
    print("--- Loading Reward Ablation Data ---")
    all_data = []
    hue_order = []

    for label, filename in FILES.items():
        path = os.path.join(LOG_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 统一筛选 GNN-DRL
                df = df[df['model'] == 'GNN-DRL'].copy()
                df['variant'] = label
                all_data.append(df)
                hue_order.append(label)
                print(f"Loaded: {label}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not all_data: return

    df_final = pd.concat(all_data, ignore_index=True)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("P1v16: Reward Component Ablation Analysis", fontsize=16, fontweight='bold')

    # 1. V2V Success
    sns.lineplot(data=df_final, x="vehicle_count", y="v2v_success_rate", hue="variant", hue_order=hue_order, style="variant", markers=True, markersize=9, linewidth=2.5, ax=axes[0])
    axes[0].set_title("V2V Reliability (Success Rate)", fontsize=14)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    # 2. V2I Capacity (重点看 R1)
    sns.lineplot(data=df_final, x="vehicle_count", y="v2i_sum_capacity_mbps", hue="variant", hue_order=hue_order, style="variant", markers=True, markersize=9, linewidth=2.5, ax=axes[1])
    axes[1].set_title("System Efficiency (V2I Capacity)", fontsize=14)
    axes[1].set_ylabel("Mbps")

    # 3. Decision Time (应该没啥区别，作为对照)
    sns.lineplot(data=df_final, x="vehicle_count", y="decision_time_ms", hue="variant", hue_order=hue_order, style="variant", markers=True, markersize=9, linewidth=2.5, ax=axes[2])
    axes[2].set_title("Inference Complexity", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE), dpi=300)
    print(f"Analysis plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    plot_analysis()