import os
import subprocess

base_dir = r"data\myExperiment"

python_path = r"C:\Users\xx\AppData\Local\conda\conda\envs\test2\python.exe"

for tf_name in os.listdir(base_dir):
    tf_dir = os.path.join(base_dir, tf_name)
    if os.path.isdir(tf_dir):
        command = f'"{python_path}" "D:\\experiment\\deepbind\\DeepBind-Pytorch\\auto_train_load_parameter.py" {tf_name}'
        print(f"command: {command}")

        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"error: {e}")