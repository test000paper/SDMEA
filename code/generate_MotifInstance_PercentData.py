import os
import random


def read_fasta(file_path):
    """
    读取FASTA文件并返回一个字典，其中包含序列名和对应的序列。
    """
    sequences = {}
    with open(file_path, 'r') as file:
        seq_name = None
        seq = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if seq_name:
                    sequences[seq_name] = ''.join(seq)
                seq_name = line[1:]
                seq = []
            else:
                seq.append(line)
        if seq_name:
            sequences[seq_name] = ''.join(seq)
    return sequences


def shuffle_by_dinucleotide(seq, k):
    """
    保持k核苷酸频率的打乱序列。将序列按k核苷酸进行分组，打乱这些k核苷酸组后重新拼接成序列。
    确保打乱后的序列与原序列长度一致，最后几个碱基也参与打乱。
    """
    # 将序列按k核苷酸进行分组
    kmers = []
    for i in range(0, len(seq) - k + 1, k):
        kmers.append(seq[i:i + k])

    # 如果序列长度不是k的倍数，保留剩余的部分并加入到打乱池中
    if len(seq) % k != 0:
        kmers.append(seq[-(len(seq) % k):])  # 单独处理剩余部分

    # 打乱k核苷酸的顺序
    random.shuffle(kmers)

    # 生成打乱后的序列
    shuffled_seq = ''.join(kmers)

    # 确保打乱后的序列长度与原序列一致
    if len(shuffled_seq) > len(seq):
        shuffled_seq = shuffled_seq[:len(seq)]  # 如果打乱后的序列过长，截取合适的长度
    elif len(shuffled_seq) < len(seq):
        shuffled_seq += shuffled_seq[:len(seq) - len(shuffled_seq)]  # 如果打乱后的序列过短，补充字符

    return shuffled_seq


def process_fasta(input_file, output_file, original_ratio, k):
    sequences = read_fasta(input_file)

    with open(output_file, 'w') as out_file:
        total_sequences = len(sequences)
        # # 控制原序列和打乱序列的数量(尽可能多地用上数据)
        # if original_ratio == 0.5:
        #     num_original = total_sequences
        #     num_shuffled = total_sequences
        # elif original_ratio < 0.5:
        #     num_original = int(total_sequences * (original_ratio / (1-original_ratio)))
        #     num_shuffled = total_sequences
        # else:  # original_ratio > 0.5
        #     num_original = total_sequences
        #     num_shuffled = int(total_sequences * ((1-original_ratio) / original_ratio))

        num_original = int(total_sequences * original_ratio)
        num_shuffled = total_sequences - num_original

        original_count = 0
        shuffled_count = 0

        for idx, (seq_name, seq) in enumerate(sequences.items(), 1):
            # 将序列转换为大写
            seq = seq.upper()
            # 打乱序列
            shuffled_seq = shuffle_by_dinucleotide(seq, k)

            if original_count < num_original:
                # 写入原始序列
                new_seq_name = f">sequence_{idx}_original"
                out_file.write(f"{new_seq_name}\n{seq}\n")
                original_count += 1

            if shuffled_count < num_shuffled:
                # 写入打乱序列
                new_seq_name = f">sequence_{idx}_shuffled"
                out_file.write(f"{new_seq_name}\n{shuffled_seq}\n")
                shuffled_count += 1

            # 如果两个序列都已写入，则跳出循环
            if original_count >= num_original and shuffled_count >= num_shuffled:
                break


def process_directory(base_directory, original_ratio, k):
    # 遍历MotifSearch_Validation目录下的所有子目录（每个子目录对应一个转录因子）
    for tf_dir in os.listdir(base_directory):
        tf_path = os.path.join(base_directory, tf_dir)
        if os.path.isdir(tf_path):
            k_path = os.path.join(tf_path, f"{k}mer")
            os.makedirs(k_path, exist_ok=True)  # 如果目录不存在则创建

            input_file = os.path.join(tf_path, f"{tf_dir}_1motifs.fa")
            if os.path.exists(input_file):
                # 根据比例动态生成输出文件名
                ratio_str = f"{int(original_ratio * 100)}percent"  # 转换为类似 "70percent"
                output_file = os.path.join(k_path, f"{tf_dir}_{k}mer_{ratio_str}.fa")
                # 处理该子目录中的FA文件
                process_fasta(input_file, output_file, original_ratio, k)
                print(f"Processed {input_file} -> {output_file}")


# 使用示例
base_directory = "MotifSearch_Validation555"  # 包含转录因子子目录的根目录
# original_ratio = 0.7  # 设置原序列占比，可以调整为0.2、0.5等值
# process_directory(base_directory, original_ratio)
max_k = 5

for k in range(1, max_k + 1):
    # 循环执行从0.1到1.0的数据，步长为0.1
    for original_ratio in [i / 10 for i in range(0, 11)]:
        process_directory(base_directory, original_ratio, k)
