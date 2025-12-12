import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# 确保这里的路径对应您 Main.py 里 global_logger 的设置
LOG_DIR = "training_results"

# === P1v18 文件名映射 (关键修改) ===
# 这里的 Key 是图例上显示的文字，Value 是 p1v18_run_ablation.py 生成的文件名
FILES = {
    "E0: Baseline (Optimal)": "scalability_P1v18_E0.csv",
    "E1: No V2I Penalty": "scalability_P1v18_E1.csv",
    "E2: No Delay Reward": "scalability_P1v18_E2.csv",
    "E3: Only SNR (Greedy)": "scalability_P1v18_E3.csv"
}

PLOT_FILE = "P1v18_Final_Ablation_Analysis.png"


def plot_analysis():
    print("--- Loading P1v18 Ablation Data ---")
    all_data = []
    hue_order = []

    # 1. 读取数据
    for label, filename in FILES.items():
        path = os.path.join(LOG_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 假设您的模型名称在 CSV 里叫 'GNN-DRL' (Main.py 里的设置)
                # 如果有多个模型列，只取 GNN 相关行
                if 'model' in df.columns:
                    # 只要包含 GNN 的行 (通常只有一个模型在跑)
                    df = df[df['model'].str.contains('GNN', case=False, na=False)].copy()

                df['variant'] = label
                all_data.append(df)
                hue_order.append(label)
                print(f"Loaded: {label} <- {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: File not found: {path}")

    if not all_data:
        print("No data found! Did you run p1v18_run_ablation.py?")
        return

    df_final = pd.concat(all_data, ignore_index=True)

    # 2. 开始绘图
    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("P1v18: Final Reward Component Ablation (Optimized Weights)", fontsize=16, fontweight='bold')

    # 子图 1: V2V 可靠性
    sns.lineplot(data=df_final, x="vehicle_count", y="v2v_success_rate", hue="variant", hue_order=hue_order,
                 style="variant", markers=True, markersize=10, linewidth=2.5, ax=axes[0])
    axes[0].set_title("V2V Reliability (Success Rate)", fontsize=14)
    axes[0].set_ylabel("Success Rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(title="Ablation Variant")

    # 子图 2: V2I 容量 (验证利他性)
    sns.lineplot(data=df_final, x="vehicle_count", y="v2i_sum_capacity_mbps", hue="variant", hue_order=hue_order,
                 style="variant", markers=True, markersize=10, linewidth=2.5, ax=axes[1])
    axes[1].set_title("System Efficiency (V2I Capacity)", fontsize=14)
    axes[1].set_ylabel("Sum Capacity (Mbps)")
    axes[1].legend(title="Ablation Variant")

    # 子图 3: P95 延迟 (验证时延敏感度) - 如果 CSV 里有 p95_delay_ms 列
    # 如果没有 P95 列，可以画 decision_time_ms 或者 mean_delay
    y_metric = "p95_delay_ms" if "p95_delay_ms" in df_final.columns else "decision_time_ms"
    y_label = "P95 Latency (ms)" if "p95_delay_ms" in df_final.columns else "Inference Time (ms)"

    sns.lineplot(data=df_final, x="vehicle_count", y=y_metric, hue="variant", hue_order=hue_order, style="variant",
                 markers=True, markersize=10, linewidth=2.5, ax=axes[2])
    axes[2].set_title(f"Latency Metric ({y_label})", fontsize=14)
    axes[2].set_ylabel(y_label)
    axes[2].legend(title="Ablation Variant")

    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, PLOT_FILE)
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Plot saved to: {plot_path}")


if __name__ == "__main__":
    plot_analysis()