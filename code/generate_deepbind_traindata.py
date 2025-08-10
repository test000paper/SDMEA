import os
import random

def shuffle_dinucleotide(sequence):
    dinucleotides = [sequence[i:i + 2] for i in range(0, len(sequence) - 1, 2)]
    random.shuffle(dinucleotides)  
    shuffled_sequence = ''.join(dinucleotides)

    if len(shuffled_sequence) < len(sequence):
        shuffled_sequence += sequence[-1]  
    return shuffled_sequence

def process_fasta(input_file, output_ac, output_b, max_sequences=1000):

    with open(input_file, 'r') as infile, open(output_ac, 'w') as ac_file, open(output_b, 'w') as b_file:
        ac_file.write("FoldID\tEventID\tseq\tBound\n")
        b_file.write("FoldID\tEventID\tseq\tBound\n")

        sequence_count = 0  
        even_sequences = []  
        for line in infile:
            if line.startswith(">"):  
                sequence_count += 1

                sequence = next(infile).strip().upper()  

                event_id = f"seq_{sequence_count:05d}_peak"

                if sequence_count % 2 == 1:  
                    ac_file.write(f"A\t{event_id}\t{sequence}\t1\n")
                else:  
                    if sequence_count <= max_sequences:  
                        b_file.write(f"A\t{event_id}\t{sequence}\t1\n")
                        even_sequences.append(sequence)  
                    else:  
                        ac_file.write(f"A\t{event_id}\t{sequence}\t1\n")

        for idx, sequence in enumerate(even_sequences):
            shuffled_sequence = shuffle_dinucleotide(sequence)
            event_id = f"seq_{(idx + 1) * 2:05d}_shuf"  
            b_file.write(f"A\t{event_id}\t{shuffled_sequence}\t0\n")

def process_folder(root_folder, output_folder):
    for tf_folder in os.listdir(root_folder):
        tf_folder_path = os.path.join(root_folder, tf_folder)
        if os.path.isdir(tf_folder_path):  
            output_tf_folder = os.path.join(output_folder, tf_folder)
            os.makedirs(output_tf_folder, exist_ok=True)

            for file_name in os.listdir(tf_folder_path):
                if file_name.endswith(".narrowPeak.fa"):
                    input_file = os.path.join(tf_folder_path, file_name)
                    output_ac = os.path.join(output_tf_folder, file_name.replace(".narrowPeak.fa", "_Stanford_AC.seq"))
                    output_b = os.path.join(output_tf_folder, file_name.replace(".narrowPeak.fa", "_Stanford_B.seq"))

                    print(f"Processing {input_file}...")
                    process_fasta(input_file, output_ac, output_b)
                    print(f"Generated {output_ac} and {output_b}")

root_folder = "690ChIP-seq_101fa"
output_folder = "690ChIP-seq_101fa_deepbind_traindata"
process_folder(root_folder, output_folder)