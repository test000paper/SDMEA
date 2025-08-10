import os
import subprocess

base_dir = r"D:\experiment\deepbind\DeepBind-Pytorch\data\myExperiment_change"
output_base_dir = r"D:\experiment\deepbind\DeepBind-Pytorch\data\input_change\比例不同数据量不变\output"  # 输出根目录
input_dir = r"D:\experiment\deepbind\DeepBind-Pytorch\data\input_change\比例不同数据量不变"  # 新的datasets目录

original_dir = os.path.join(input_dir, "original")
shuffled_dir = os.path.join(input_dir, "shuffled")

tf_names = []
model_names = []


for tf_name in os.listdir(original_dir):
    tf_dir = os.path.join(original_dir, tf_name)
    if os.path.isdir(tf_dir):
        tf_names.append(tf_name)

for model_name in os.listdir(base_dir):
    model_dir = os.path.join(base_dir, model_name)
    if os.path.isdir(model_dir):
        model_names.append(model_name)


for tf in tf_names:
    tf_dir = os.path.join(original_dir, tf)

    tf_output_dir = os.path.join(output_base_dir, tf)
    os.makedirs(tf_output_dir, exist_ok=True)

    for input_file in os.listdir(tf_dir):
        if input_file.endswith(".fa"):
            percent = input_file.split('_')[-1].replace('.fa', '')

            percent_output_dir = os.path.join(tf_output_dir, percent)
            os.makedirs(percent_output_dir, exist_ok=True)

            output_target_dir = os.path.join(percent_output_dir, "target")
            os.makedirs(output_target_dir, exist_ok=True)

            output_background_dir = os.path.join(percent_output_dir, "background")
            os.makedirs(output_background_dir, exist_ok=True)

            input_file_path = os.path.join(tf_dir, input_file)

            background_file = input_file.replace(".fa", "_shuffled.fa")
            background_file_path = os.path.join(shuffled_dir, tf, background_file)

            for model_name in model_names:
                model_path = os.path.join(base_dir, model_name, "model", "MyModel_2.pth")

                output_target_file = os.path.join(
                    output_target_dir,
                    f"{tf}_prediction_on_{model_name}.txt"
                )

                output_background_file = os.path.join(
                    output_background_dir,
                    f"{tf}_shuffled_prediction_on_{model_name}.txt"
                )

                cmd_target = [
                    'python', 'only_prediction_withParameter.py',
                    '--tf_name', model_name,
                    '--model_path', model_path,
                    '--fasta_file', input_file_path,
                    '--output_file', output_target_file
                ]

                cmd_background = [
                    'python', 'only_prediction_withParameter.py',
                    '--tf_name', model_name,
                    '--model_path', model_path,
                    '--fasta_file', background_file_path,
                    '--output_file', output_background_file
                ]

                # 执行预测
                try:
                    subprocess.run(cmd_target, check=True)
                    print(f"{tf}/{percent} 目标序列在 {model_name} 模型预测完成")
                except subprocess.CalledProcessError as e:
                    print(f"{tf}/{percent} 目标序列在 {model_name} 模型预测出错：{e}")

                try:
                    subprocess.run(cmd_background, check=True)
                    print(f"{tf}/{percent} 背景序列在 {model_name} 模型预测完成")
                except subprocess.CalledProcessError as e:
                    print(f"{tf}/{percent} 背景序列在 {model_name} 模型预测出错：{e}")
