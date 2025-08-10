import os
import pandas as pd


def get_motif_rank(statistics_file):
    df = pd.read_csv(statistics_file, sep='\t')
    df['z_score_rank'] = df['z_score'].rank(ascending=False, method='min')
    return df[['motif_id', 'z_score', 'z_score_rank']]


root_dir = r'data\input\不同背景信息数据\output'

size_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

rank_results = []
target_motif = "on_CEBPD"

for size_dir in size_dirs:
    size = os.path.basename(size_dir)
    statistics_file = os.path.join(size_dir, f'{size}_motif_statistics.txt')

    if os.path.exists(statistics_file):
        motif_ranks = get_motif_rank(statistics_file)
        row = motif_ranks.loc[motif_ranks['motif_id'] == target_motif]

        if not row.empty:
            z_score = row['z_score'].values[0]
            rank_results.append(f"{size}: on_CEBPD z_score = {z_score}")


for result in rank_results:
    print(result)
