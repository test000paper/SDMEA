# 定义文件路径
tab_file_path = 'FIMOresult/MA0003.5_extracted.fa.tab'  
all_sequences_file = 'FIMOresult/MA0003.5_extracted.fa'  

with open(tab_file_path, 'r') as tab_file:
    header = tab_file.readline().strip().split('\t')  
    motif_names = header[1:]  

target_sequences = {motif: set() for motif in motif_names}

with open(tab_file_path, 'r') as tab_file:
    lines = tab_file.readlines()
    for line in lines[1:]:  
        parts = line.strip().split('\t')
        sequence_name = parts[0]  
        for i, count in enumerate(parts[1:]):  
            if int(count) == 0:  
                target_sequences[motif_names[i]].add(sequence_name)  

for motif, sequences in target_sequences.items():
    output_file = f'result/extracted_filtered/{motif}_extracted_filtered.fa'  
    with open(all_sequences_file, 'r') as input_file, open(output_file, 'w') as output_file:
        write_sequence = False  
        for line in input_file:
            if line.startswith('>'):  
                sequence_id = line.strip()[1:]  
                if sequence_id in sequences:  
                    write_sequence = True  
                    output_file.write(line)  
                else:
                    write_sequence = False  
            elif write_sequence:  
                output_file.write(line)  
    print(f"Succesfully，{motif} 's reslut save to {output_file}")