import os
import gzip
from pyfaidx import Fasta


def process_narrowpeak_files(input_dir, genome_file, output_dir):
    genome = Fasta(genome_file)

    for root, dirs, files in os.walk(input_dir):
        transcription_factor = os.path.basename(root)
        print(f"Processing transcription factor: {transcription_factor}")

        for file in files:
            if file.endswith(".narrowPeak.gz"):  
                narrowpeak_path = os.path.join(root, file)

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
                peak_offset = int(fields[9])  

                peak_position = start + peak_offset

                new_start = peak_position - 50
                new_end = peak_position + 51

                try:
                    sequence = genome[chrom][new_start:new_end].seq
                    fa_file.write(f">{chrom}:{new_start}-{new_end}\n{sequence}\n")
                except KeyError:
                    print(f"Warning: {chrom}:{new_start}-{new_end} not found in genome file.")
    except EOFError:
        print(f"Error: File {narrowpeak_path} is corrupted. Skipping...")


if __name__ == "__main__":
    input_dir = r"690ChIP-seq-processed"
    genome_file = r"data\MEA\male.hg19.fa"
    output_dir = r"690ChIP-seq_101fa"

    process_narrowpeak_files(input_dir, genome_file, output_dir)