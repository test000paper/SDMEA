import os
import gzip
from pyfaidx import Fasta


def process_narrowpeak_files(input_dir, genome_file, output_dir):
    # 加载基因组文件
    genome = Fasta(genome_file)

    # 遍历输入目录中的文件夹和文件
    for root, dirs, files in os.walk(input_dir):
        # 打印当前处理的目录（即当前转录因子）
        transcription_factor = os.path.basename(root)
        print(f"Processing transcription factor: {transcription_factor}")

        for file in files:
            if file.endswith(".narrowPeak.gz"):  # 找到 narrowPeak 文件
                narrowpeak_path = os.path.join(root, file)

                # 计算输出路径，保持目录结构
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                output_file = os.path.join(output_folder, os.path.splitext(file)[0] + ".fa")
                extract_sequences(narrowpeak_path, genome, output_file)


def extract_sequences(narrowpeak_path, genome, output_file):
    try:
        with gzip.open(narrowpeak_path, 'rt') as np_file, open(output_file, 'w') as fa_file:
            for line in np_file:
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue

                chrom = fields[0]
                start = int(fields[1])
                peak_offset = int(fields[9])  # 第10列是峰值偏移位置

                # 计算峰值位置
                peak_position = start + peak_offset

                # 计算新的起始和终止位置，以峰值位置为中心截取101bp
                new_start = peak_position - 50
                new_end = peak_position + 51

                # 提取序列
                try:
                    sequence = genome[chrom][new_start:new_end].seq
                    fa_file.write(f">{chrom}:{new_start}-{new_end}\n{sequence}\n")
                except KeyError:
                    print(f"Warning: {chrom}:{new_start}-{new_end} not found in genome file.")
    except EOFError:
        print(f"Error: File {narrowpeak_path} is corrupted. Skipping...")


if __name__ == "__main__":
    # 输入和输出路径
    input_dir = r"D:\资料\模体\交接工作\交接胡雅娜师姐论文工作\代码\meirlop模体富集论文原文代码\meirlop-master\690ChIP-seq-processed"
    genome_file = r"D:\资料\模体\交接工作\交接胡雅娜师姐论文工作\代码\meirlop模体富集论文原文代码\meirlop-master\data\male.hg19.fa"
    output_dir = r"D:\资料\模体\交接工作\交接胡雅娜师姐论文工作\代码\meirlop模体富集论文原文代码\meirlop-master\690ChIP-seq_101fa"

    # 执行
    process_narrowpeak_files(input_dir, genome_file, output_dir)