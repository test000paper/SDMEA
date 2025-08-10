import os
import re
import requests

# 输入文件路径
input_file = r"D:\资料\模体\交接工作\交接胡雅娜师姐论文工作\代码\meirlop模体富集论文原文代码\meirlop-master\input_motifs.txt"
output_folder = r"D:\资料\模体\交接工作\交接胡雅娜师姐论文工作\代码\meirlop模体富集论文原文代码\meirlop-master\motif_bed"
base_url = "https://jaspar.elixir.no/download/data/2024/bed/"

# 从文件中读取模体 ID 和 TF 名称
def read_motif_ids(file_path):
    motif_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith(">"):
                match = re.match(r">(\S+)\s+(\S+)", line)
                if match:
                    motif_id = match.group(1)
                    tf_name = match.group(2)
                    motif_data.append((motif_id, tf_name))
    return motif_data

# 下载 BED 文件并保存
def download_bed_file(motif_id, output_dir):
    url = f"{base_url}{motif_id}.bed"
    output_path = os.path.join(output_dir, f"{motif_id}.bed")

    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            # 保存到文件
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# 主函数
def main():
    # 读取模体数据
    motif_data = read_motif_ids(input_file)
    print(f"Found {len(motif_data)} motifs in the file.")

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 下载每个模体的 BED 文件
    for motif_id, tf_name in motif_data:
        download_bed_file(motif_id, output_folder)

if __name__ == "__main__":
    main()
