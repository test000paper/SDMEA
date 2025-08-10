import os
import random


def read_fasta(file_path):

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

    kmers = []
    for i in range(0, len(seq) - k + 1, k):
        kmers.append(seq[i:i + k])

    if len(seq) % k != 0:
        kmers.append(seq[-(len(seq) % k):])  

    random.shuffle(kmers)

    shuffled_seq = ''.join(kmers)

    if len(shuffled_seq) > len(seq):
        shuffled_seq = shuffled_seq[:len(seq)]  
    elif len(shuffled_seq) < len(seq):
        shuffled_seq += shuffled_seq[:len(seq) - len(shuffled_seq)]  

    return shuffled_seq


def process_fasta(input_file, output_file, original_ratio, k):
    sequences = read_fasta(input_file)

    with open(output_file, 'w') as out_file:
        total_sequences = len(sequences)

        num_original = int(total_sequences * original_ratio)
        num_shuffled = total_sequences - num_original

        original_count = 0
        shuffled_count = 0

        for idx, (seq_name, seq) in enumerate(sequences.items(), 1):
            seq = seq.upper()
            shuffled_seq = shuffle_by_dinucleotide(seq, k)

            if original_count < num_original:
                new_seq_name = f">sequence_{idx}_original"
                out_file.write(f"{new_seq_name}\n{seq}\n")
                original_count += 1

            if shuffled_count < num_shuffled:
                new_seq_name = f">sequence_{idx}_shuffled"
                out_file.write(f"{new_seq_name}\n{shuffled_seq}\n")
                shuffled_count += 1

            if original_count >= num_original and shuffled_count >= num_shuffled:
                break


def process_directory(base_directory, original_ratio, k):
    for tf_dir in os.listdir(base_directory):
        tf_path = os.path.join(base_directory, tf_dir)
        if os.path.isdir(tf_path):
            k_path = os.path.join(tf_path, f"{k}mer")
            os.makedirs(k_path, exist_ok=True)  

            input_file = os.path.join(tf_path, f"{tf_dir}_1motifs.fa")
            if os.path.exists(input_file):
                ratio_str = f"{int(original_ratio * 100)}percent"  
                output_file = os.path.join(k_path, f"{tf_dir}_{k}mer_{ratio_str}.fa")
                process_fasta(input_file, output_file, original_ratio, k)
                print(f"Processed {input_file} -> {output_file}")


base_directory = "MotifSearch_Validation"  
# original_ratio = 0.7  
# process_directory(base_directory, original_ratio)


max_k = 5
for k in range(1, max_k + 1):
    for original_ratio in [i / 10 for i in range(0, 11)]:
        process_directory(base_directory, original_ratio, k)
