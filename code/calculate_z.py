import os
import pandas as pd
import numpy as np


def compute_motif_statistics(target_file, background_file):

    target_df = pd.read_csv(target_file, sep='\t')
    background_df = pd.read_csv(background_file, sep='\t')

    motifs = target_df.columns[1:]
    results = []

    for motif in motifs:
        A = target_df[motif].mean()
        B = background_df[motif].mean()
        C = background_df[motif].std()
        z_score = (A - B) / C if C != 0 else np.nan

        target_quartiles = np.percentile(target_df[motif].dropna(), [25, 50, 75])
        background_quartiles = np.percentile(background_df[motif].dropna(), [25, 50, 75])

        results.append([
            motif, A, B, C, z_score,
            target_quartiles[0], target_quartiles[1], target_quartiles[2],
            background_quartiles[0], background_quartiles[1], background_quartiles[2],
            target_quartiles[1] - background_quartiles[0]
        ])

    results_df = pd.DataFrame(results, columns=[
        "motif_id", "observed_mean", "expected_mean", "expected_sd", "z_score",
        "target_25%", "target_50%", "target_75%",
        "background_25%", "background_50%", "background_75%",
        "diff_50%-25%"
    ])
    return results_df


# 定义根目录
root_dir = r'D:\experiment\deepbind\DeepBind-Pytorch\data\input\比例变化数据量不变\output'

# 获取所有数据量子目录
size_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for size_dir in size_dirs:
    size = os.path.basename(size_dir)
    target_file = os.path.join(size_dir, f'{size}_target_merged.txt')
    background_file = os.path.join(size_dir, f'{size}_background_merged.txt')

    if os.path.exists(target_file) and os.path.exists(background_file):
        results_df = compute_motif_statistics(target_file, background_file)
        output_file = os.path.join(size_dir, f'{size}_motif_statistics.txt')
        results_df.to_csv(output_file, sep='\t', index=False)
        print(f"统计结果已保存到: {output_file}")