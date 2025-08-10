import os
import random


def dinucleotide_shuffle_sequence(sequence, seed=12345):

    random.seed(seed)
    sequence = sequence.upper()

    dinucleotide_pairs = [sequence[i:i + 2] for i in range(0, len(sequence) - 1, 2)]

    if len(sequence) % 2 == 1:
        dinucleotide_pairs.append(sequence[-1])

    random.shuffle(dinucleotide_pairs)

    shuffled_sequence = ''.join(dinucleotide_pairs)
    return shuffled_sequence


def shuffle_txt_sequences(input_txt, output_txt, seed=12345):
    with open(input_txt, "r") as input_handle, open(output_txt, "w") as output_handle:
        sequence_name = ""
        for line in input_handle:
            line = line.strip()
            if line.startswith(">"):
                sequence_name = line
                output_handle.write(f"{sequence_name}\n")
            elif line:
                shuffled_sequence = dinucleotide_shuffle_sequence(line, seed)
                output_handle.write(f"{shuffled_sequence}\n")


input_dir = "data/input_change/比例不同数据量不变/original"
output_dir = "data/input_change/比例不同数据量不变/shuffled"


for tf_dir in os.listdir(input_dir):
    tf_dir_path = os.path.join(input_dir, tf_dir)
    if os.path.isdir(tf_dir_path):
        output_tf_dir = os.path.join(output_dir, tf_dir)
        os.makedirs(output_tf_dir, exist_ok=True)

        for file_name in os.listdir(tf_dir_path):
            if file_name.endswith(".fa"):
                input_path = os.path.join(tf_dir_path, file_name)

                output_file_name = file_name.replace(".fa", "_shuffled.fa")
                output_path = os.path.join(output_tf_dir, output_file_name)

                shuffle_txt_sequences(input_path, output_path)
                print(f"打乱后的序列已保存到: {output_path}")

