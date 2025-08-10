import os
from Bio import SeqIO
from Bio.Seq import Seq


def get_genome_file(bed_file):
    """
    根据BED文件的第四列信息，返回相应的基因组FASTA文件路径
    """
    with open(bed_file, "r") as bed:
        first_line = bed.readline().strip()
        genome_info = first_line.split("\t")[3]  # 第四列
        if "hg38" in genome_info:
            return "data/hg38.fa"
        elif "hg19" in genome_info:
            return "data/hg19.fa"
        else:
            print(f"Warning: Unsupported genome version found in {genome_info}.")
            return None  # 返回None表示不支持


def modify_bed_regions(input_file):
    """
    计算基于中心位置向左右各扩展50bp的新BED区间，但不保存文件
    """
    modified_regions = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            chr_name = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3]
            strand = fields[5]

            # 计算中心位置
            center = start + (end - start) // 2
            new_start = max(center - 50, 0)
            new_end = center + 51

            modified_regions.append((chr_name, new_start, new_end, name, strand))

    return modified_regions


def extract_sequences(fasta_file, modified_regions, output_file):
    """
    根据修改后的BED区域从FASTA文件中提取序列
    """
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    with open(output_file, "w") as out:
        sequence_count = 1
        prev_start = prev_end = prev_chrom = None

        for chrom, start, end, name, strand in modified_regions:
            if chrom not in genome:
                print(f"Warning: Chromosome {chrom} not found in the FASTA file.")
                continue

            # if prev_chrom == chrom and prev_start == start and prev_end == end:
            #     continue

            # prev_start, prev_end, prev_chrom = start, end, chrom
            sequence = genome[chrom].seq[start:end]

            if strand == "-":
                sequence = sequence.reverse_complement()

            out.write(f">sequence_{sequence_count}:{start}-{end}\n{str(sequence).upper()}\n")
            sequence_count += 1


if __name__ == "__main__":
    bed_file = "690ChIP-seq_motif_bed/ATF2/MA1632.2.bed"  # 固定的 BED 文件
    output_dir = "模体富集实验/单独提取模体实例"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    genome_file = get_genome_file(bed_file)
    if genome_file:
        modified_regions = modify_bed_regions(bed_file)
        output_fasta = os.path.join(output_dir, "ATF2_sequences1.fa")
        extract_sequences(genome_file, modified_regions, output_fasta)
        print(f"FASTA sequences saved to: {output_fasta}")

# if __name__ == "__main__":
#     input_dir = "690ChIP-seq_motif_bed"  # 输入目录
#     output_base_dir = "MotifSearch_Validation222"  # 输出目录
#     os.makedirs(output_base_dir, exist_ok=True)
#
#     for tf_name in os.listdir(input_dir):
#         tf_dir = os.path.join(input_dir, tf_name)
#         if os.path.isdir(tf_dir):
#             bed_files = [f for f in os.listdir(tf_dir) if f.endswith(".bed")]
#             for bed_file in bed_files:
#                 bed_path = os.path.join(tf_dir, bed_file)
#                 genome_file = get_genome_file(bed_path)
#                 if genome_file:
#                     modified_regions = modify_bed_regions(bed_path)
#                     output_tf_dir = os.path.join(output_base_dir, tf_name)
#                     os.makedirs(output_tf_dir, exist_ok=True)
#                     output_fasta = os.path.join(output_tf_dir, f"{tf_name}_sequence.fa")
#                     extract_sequences(genome_file, modified_regions, output_fasta)
#                     print(f"FASTA sequences saved to: {output_fasta}")


# import random
#
# NUCLEOTIDES = ["A", "T", "C", "G"]  # 可能的碱基
#
#
# def mutate_fasta(input_fa, output_fa):
#     """
#     读取FASTA文件，并将每个序列的第47~55个碱基替换为随机碱基
#     """
#     with open(input_fa, "r") as infile, open(output_fa, "w") as outfile:
#         seq_name = None
#         seq_content = []
#
#         for line in infile:
#             if line.startswith(">"):  # 序列名行
#                 if seq_name and seq_content:
#                     mutated_seq = mutate_sequence("".join(seq_content))
#                     outfile.write(f"{seq_name}\n{mutated_seq}\n")
#
#                 seq_name = line.strip()
#                 seq_content = []
#             else:
#                 seq_content.append(line.strip())
#
#         # 处理最后一个序列
#         if seq_name and seq_content:
#             mutated_seq = mutate_sequence("".join(seq_content))
#             outfile.write(f"{seq_name}\n{mutated_seq}\n")
#
#     print(f"Mutated FASTA file saved to: {output_fa}")
#
#
# def mutate_sequence(sequence):
#     """
#     对序列的第47~55个碱基进行随机替换
#     """
#     seq_list = list(sequence)  # 转换为列表以便修改
#     start, end = 46, 55  # Python索引从0开始，第47位对应索引46
#
#     if len(seq_list) > end:
#         for i in range(start, end + 1):
#             seq_list[i] = random.choice(NUCLEOTIDES)
#
#     return "".join(seq_list)
#
#
# if __name__ == "__main__":
#     input_fasta = f"模体富集实验\纯背景序列\CEBPD_single.fa"  # 你的输入FASTA文件路径
#     output_fasta = f"模体富集实验\纯背景序列\CEBPD_lack.fa"  # 你的输出FASTA文件路径
#
#     mutate_fasta(input_fasta, output_fasta)
