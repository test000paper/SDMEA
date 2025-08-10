import os
from Bio import SeqIO
from Bio.Seq import Seq


def get_genome_file(bed_file):

    with open(bed_file, "r") as bed:
        first_line = bed.readline().strip()
        genome_info = first_line.split("\t")[3]  # 第四列
        if "hg38" in genome_info:
            return "data/hg38.fa"
        elif "hg19" in genome_info:
            return "data/hg19.fa"
        else:
            print(f"Warning: Unsupported genome version found in {genome_info}.")
            return None  


def modify_bed_regions(input_file):
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

            center = start + (end - start) // 2
            new_start = max(center - 50, 0)
            new_end = center + 51

            modified_regions.append((chr_name, new_start, new_end, name, strand))

    return modified_regions


def extract_sequences(fasta_file, modified_regions, output_file):

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
    bed_file = "690ChIP-seq_motif_bed/ATF2/MA1632.2.bed"  
    output_dir = "experiment/MotifSearch_Validation"  
    os.makedirs(output_dir, exist_ok=True)

    genome_file = get_genome_file(bed_file)
    if genome_file:
        modified_regions = modify_bed_regions(bed_file)
        output_fasta = os.path.join(output_dir, "ATF2_sequences1.fa")
        extract_sequences(genome_file, modified_regions, output_fasta)
        print(f"FASTA sequences saved to: {output_fasta}")


