import os
import glob
import re

root_dir = r'D:\experiment\deepbind\DeepBind-Pytorch\data\input\手动植入多模体实例的序列\output'

sub_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for sub_dir in sub_dirs:
    for category in ['background', 'target']:
        category_dir = os.path.join(sub_dir, category)
        if not os.path.exists(category_dir):
            continue

        merged_data = {}
        output_file = os.path.join(sub_dir, f'{os.path.basename(sub_dir)}_{category}_merged.txt')

        txt_files = glob.glob(os.path.join(category_dir, "*.txt"))

        tf_names = [re.search(r'on_(.*?)\.txt', os.path.basename(f)).group(1) if re.search(r'on_(.*?)\.txt',os.path.basename(f)) else os.path.basename(f).replace('.txt', '') for f in txt_files]

        for txt_file, tf_name in zip(txt_files, tf_names):
            with open(txt_file, 'r') as file:
                for line in file:
                    sequence, value = line.strip().split('\t')
                    if sequence not in merged_data:
                        merged_data[sequence] = {}
                    merged_data[sequence][tf_name] = value

        headers = ['event'] + tf_names
        merged_lines = []
        for sequence, values in merged_data.items():
            row = [sequence] + [values.get(tf, '0') for tf in headers[1:]]
            merged_lines.append(row)

        with open(output_file, 'w') as f:
            f.write('\t'.join(headers) + '\n')
            for row in merged_lines:
                f.write('\t'.join(row) + '\n')

        print(f"合并后的文件已保存到: {output_file}")


