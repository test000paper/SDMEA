import os
import subprocess


base_dir = r"data\input\myExperiment"
output_base_dir = r"data\output"  # 输出根目录

tf_names = []
model_names = []

for tf_name in os.listdir(base_dir):
    tf_dir = os.path.join(base_dir, tf_name)
    if os.path.isdir(tf_dir):
        tf_names.append(tf_name)

        output_tf_dir = os.path.join(output_base_dir, tf_name)
        os.makedirs(output_tf_dir, exist_ok=True)

model_names = tf_names

for tf_name in tf_names:
    tf_dir = os.path.join(base_dir, tf_name)

    if os.path.isdir(tf_dir) and os.path.exists(os.path.join(tf_dir, 'model')) and os.path.exists(
            os.path.join(tf_dir, f"{tf_name}_50percent.fa")) and os.path.exists(
            os.path.join(tf_dir, f"{tf_name}_shuffled.fa")):

        fasta_file = os.path.join(tf_dir, f"{tf_name}_50percent.fa")
        shuffled_file = os.path.join(tf_dir, f"{tf_name}_shuffled.fa")

        for model_name in model_names:

            model_path = model_path = os.path.join(base_dir, f"{model_name}\\model\\MyModel_2.pth")

            output_target_path = os.path.join(output_base_dir, f"{tf_name}\\target")
            os.makedirs(output_target_path, exist_ok=True)
            output_target_file = os.path.join(output_target_path, f"{tf_name}_target_predictions_on_{model_name}.txt")
            output_background_path = os.path.join(output_base_dir, f"{tf_name}\\background")
            os.makedirs(output_background_path, exist_ok=True)
            output_background_file = os.path.join(output_background_path, f"{tf_name}_background_predictions_on_{model_name}.txt")

            cmd_target = [
                'python', 'only_prediction_withParameter.py',
                '--tf_name', model_name,
                '--model_path', model_path,
                '--fasta_file', fasta_file,
                '--output_file', output_target_file
            ]

            try:
                subprocess.run(cmd_target, check=True)
                print(f"{tf_name} target sequence on {model_name} SD Model's predict finished")
            except subprocess.CalledProcessError as e:
                print(f"{tf_name} target sequence on {model_name} SD Model's predict error：{e}")

            cmd_background = [
                'python', 'only_prediction_withParameter.py',
                '--tf_name', model_name,
                '--model_path', model_path,
                '--fasta_file', shuffled_file,
                '--output_file', output_background_file
            ]

            try:
                subprocess.run(cmd_background, check=True)
                print(f"{tf_name} background sequence on {model_name} SD Model's predict finished")
            except subprocess.CalledProcessError as e:
                print(f"{tf_name} background sequence on {model_name} SD Model's predict error：{e}")
