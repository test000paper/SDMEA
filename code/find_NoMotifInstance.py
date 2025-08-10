# 定义文件路径
tab_file_path = 'result/MA0003.5_extracted.fa.tab'  # .tab 文件路径
all_sequences_file = 'result/MA0003.5_extracted.fa'  # 包含所有序列内容的文件

# 读取.tab文件的标题行，获取模体名称
with open(tab_file_path, 'r') as tab_file:
    header = tab_file.readline().strip().split('\t')  # 读取标题行并分割
    motif_names = header[1:]  # 第一列是序列名，从第二列开始是模体名称

# 为每个模体创建目标序列集合
target_sequences = {motif: set() for motif in motif_names}

# 读取.tab文件，提取每个模体的目标序列号
with open(tab_file_path, 'r') as tab_file:
    lines = tab_file.readlines()
    for line in lines[1:]:  # 跳过标题行
        parts = line.strip().split('\t')
        sequence_name = parts[0]  # 第一列是序列名
        for i, count in enumerate(parts[1:]):  # 从第二列开始是模体计数
            if int(count) == 0:  # 如果模体计数为0
                target_sequences[motif_names[i]].add(sequence_name)  # 添加到对应模体的目标集合中

# 从所有序列文件中提取目标序列，并保存到不同的输出文件
for motif, sequences in target_sequences.items():
    output_file = f'result/extracted_filtered/{motif}_extracted_filtered.fa'  # 输出文件路径
    with open(all_sequences_file, 'r') as input_file, open(output_file, 'w') as output_file:
        write_sequence = False  # 标记是否写入当前序列
        for line in input_file:
            if line.startswith('>'):  # 检查是否是序列头
                sequence_id = line.strip()[1:]  # 去掉 '>' 并提取序列号
                if sequence_id in sequences:  # 如果序列号在目标集合中
                    write_sequence = True  # 标记为需要写入
                    output_file.write(line)  # 写入序列头
                else:
                    write_sequence = False  # 否则不写入
            elif write_sequence:  # 如果需要写入当前序列
                output_file.write(line)  # 写入序列内容
    print(f"处理完成，{motif} 的结果已保存到 {output_file}")