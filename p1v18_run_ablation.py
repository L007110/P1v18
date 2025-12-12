import os
import time
import sys

# P1v18: 最终消融实验序列
# E0: 最佳基准 (1.0, 0.5, 0.1)
# E1: 去掉 V2I 惩罚
# E2: 去掉 Delay 奖励
# E3: 仅保留 SNR (贪婪)
MODES = ["E0", "E1", "E2", "E3"]
SEED = 11
EPOCHS = 1000


def run_ablation():
    # 获取当前解释器
    py_cmd = f'"{sys.executable}"'

    if not os.path.exists("training_results"):
        os.makedirs("training_results")

    print(f"=== P1v18 Final Ablation Start ({len(MODES)} jobs) ===")
    print(f"Using Python: {py_cmd}")

    for idx, mode in enumerate(MODES):
        print(f"\n>>> [Job {idx + 1}/{len(MODES)}] Running Mode: {mode} <<<")

        # 1. Training
        print(f"  Phase 1: Training ({EPOCHS} epochs)...")
        train_cmd = (
            f"{py_cmd} Main.py --run_mode TRAIN "
            f"--seed {SEED} --epochs {EPOCHS} "
            f"--ablation_mode {mode}"
        )

        ret = os.system(train_cmd)
        if ret != 0:
            print(f"!!! Error in training {mode}. Skipping.")
            continue

        # 2. Rename Log
        # 生成的文件名 Main.py 会自动设为: _P1v18_Ex
        suffix = f"_P1v18_{mode}"
        src_log = "training_results/global_metrics.csv"
        dst_log = f"training_results/convergence{suffix}_seed{SEED}.csv"

        if os.path.exists(src_log):
            if os.path.exists(dst_log): os.remove(dst_log)
            os.rename(src_log, dst_log)
            print(f"  > Log saved: {dst_log}")

        # 3. Testing
        print(f"  Phase 2: Scalability Test...")
        test_cmd = (
            f"{py_cmd} Main.py --run_mode TEST "
            f"--seed {SEED} --ablation_mode {mode}"
        )
        os.system(test_cmd)

        print(f">>> Mode {mode} Complete.")
        time.sleep(2)

    print("\n=== P1v18 All Done! ===")


if __name__ == "__main__":
    run_ablation()