import os
import random

def shuffle_dinucleotide(sequence):
    """打乱序列的二核苷酸频率，同时保持序列长度不变"""
    # 将序列分成二核苷酸片段
    dinucleotides = [sequence[i:i + 2] for i in range(0, len(sequence) - 1, 2)]
    random.shuffle(dinucleotides)  # 打乱二核苷酸顺序
    # 拼接打乱后的二核苷酸片段
    shuffled_sequence = ''.join(dinucleotides)
    # 如果长度不足 101bp，补充最后一个碱基
    if len(shuffled_sequence) < len(sequence):
        shuffled_sequence += sequence[-1]  # 补充最后一个碱基
    return shuffled_sequence

def process_fasta(input_file, output_ac, output_b, max_sequences=1000):
    # 打开输入文件和输出文件
    with open(input_file, 'r') as infile, open(output_ac, 'w') as ac_file, open(output_b, 'w') as b_file:
        # 写入表头
        ac_file.write("FoldID\tEventID\tseq\tBound\n")
        b_file.write("FoldID\tEventID\tseq\tBound\n")

        sequence_count = 0  # 记录当前处理的序列号
        even_sequences = []  # 保存前 1000 序列中的偶数序列
        for line in infile:
            if line.startswith(">"):  # 如果是序列名行
                sequence_count += 1

                # 读取序列内容行
                sequence = next(infile).strip().upper()  # 将序列内容转为大写

                # 生成 EventID
                event_id = f"seq_{sequence_count:05d}_peak"

                # 根据序列号的奇偶性写入不同的文件
                if sequence_count % 2 == 1:  # 奇数序列
                    ac_file.write(f"A\t{event_id}\t{sequence}\t1\n")
                else:  # 偶数序列
                    if sequence_count <= max_sequences:  # 前 1000 序列中的偶数序列
                        b_file.write(f"A\t{event_id}\t{sequence}\t1\n")
                        even_sequences.append(sequence)  # 保存偶数序列
                    else:  # 1000 以后的序列
                        ac_file.write(f"A\t{event_id}\t{sequence}\t1\n")

        # 将偶数序列的二核苷酸频率打乱并附加到 xxx_Stanford_B.seq
        for idx, sequence in enumerate(even_sequences):
            shuffled_sequence = shuffle_dinucleotide(sequence)
            event_id = f"seq_{(idx + 1) * 2:05d}_shuf"  # 生成打乱序列的 EventID
            b_file.write(f"A\t{event_id}\t{shuffled_sequence}\t0\n")

def process_folder(root_folder, output_folder):
    # 遍历根文件夹下的所有子文件夹
    for tf_folder in os.listdir(root_folder):
        tf_folder_path = os.path.join(root_folder, tf_folder)
        if os.path.isdir(tf_folder_path):  # 确保是文件夹
            # 在输出文件夹中创建对应的子文件夹
            output_tf_folder = os.path.join(output_folder, tf_folder)
            os.makedirs(output_tf_folder, exist_ok=True)

            # 查找子文件夹中的 .narrowPeak.fa 文件
            for file_name in os.listdir(tf_folder_path):
                if file_name.endswith(".narrowPeak.fa"):
                    input_file = os.path.join(tf_folder_path, file_name)
                    output_ac = os.path.join(output_tf_folder, file_name.replace(".narrowPeak.fa", "_Stanford_AC.seq"))
                    output_b = os.path.join(output_tf_folder, file_name.replace(".narrowPeak.fa", "_Stanford_B.seq"))

                    # 处理 FASTA 文件
                    print(f"Processing {input_file}...")
                    process_fasta(input_file, output_ac, output_b)
                    print(f"Generated {output_ac} and {output_b}")

# 调用函数处理文件夹
root_folder = "690ChIP-seq_101fa"
output_folder = "690ChIP-seq_101fa_deepbind_traindata"
process_folder(root_folder, output_folder)