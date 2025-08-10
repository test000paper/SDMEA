import os
import requests

# 文件路径和目标根目录
files_txt_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/files.txt"
target_dir = os.path.join(os.getcwd(), '690ChIP-seq')

# 下载并读取files.txt
response = requests.get(files_txt_url)
files_txt = response.text

# 按行分割files.txt内容
lines = files_txt.strip().split('\n')

# 遍历每一行
for line in lines:
    # 提取文件名和转录因子信息
    columns = line.split('\t')
    file_name = columns[0]
    metadata = columns[1]

    # 按 ';' 分隔metadata，提取第7个元素并去掉'antibody='部分
    metadata_parts = metadata.split(';')
    tf_info = metadata_parts[7].split('=')[1]  # 第7个元素为'antibody=CTCF'，取 'CTCF'

    # 创建以转录因子为名的文件夹
    target_path = os.path.join(target_dir, tf_info)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 下载文件并保存到对应的目录
    file_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/{file_name}"
    file_path = os.path.join(target_path, file_name)

    print(f"Downloading {file_name} to {file_path}")
    file_response = requests.get(file_url)

    # 确保下载成功
    if file_response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(file_response.content)
        print(f"Downloaded {file_name} successfully!")
    else:
        print(f"Failed to download {file_name}. Status code: {file_response.status_code}")