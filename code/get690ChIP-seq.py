import os
import requests

files_txt_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/files.txt"
target_dir = os.path.join(os.getcwd(), '690ChIP-seq')

response = requests.get(files_txt_url)
files_txt = response.text

lines = files_txt.strip().split('\n')

for line in lines:
    columns = line.split('\t')
    file_name = columns[0]
    metadata = columns[1]

    metadata_parts = metadata.split(';')
    tf_info = metadata_parts[7].split('=')[1]  

    target_path = os.path.join(target_dir, tf_info)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    file_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/{file_name}"
    file_path = os.path.join(target_path, file_name)

    print(f"Downloading {file_name} to {file_path}")
    file_response = requests.get(file_url)

    if file_response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(file_response.content)
        print(f"Downloaded {file_name} successfully!")
    else:
        print(f"Failed to download {file_name}. Status code: {file_response.status_code}")