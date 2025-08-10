import os
import re
import requests

# 输入文件路径
input_file = r"input_motifs.txt"
output_folder = r"motif_bed"
base_url = "https://jaspar.elixir.no/download/data/2024/bed/"


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


def download_bed_file(motif_id, output_dir):
    url = f"{base_url}{motif_id}.bed"
    output_path = os.path.join(output_dir, f"{motif_id}.bed")

    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    motif_data = read_motif_ids(input_file)
    print(f"Found {len(motif_data)} motifs in the file.")

    os.makedirs(output_folder, exist_ok=True)

    for motif_id, tf_name in motif_data:
        download_bed_file(motif_id, output_folder)

if __name__ == "__main__":
    main()
