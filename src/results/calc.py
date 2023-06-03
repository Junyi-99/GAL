import os
import re
from datetime import datetime

# 定义文件夹路径
folder_path = "./"

# 遍历文件夹中的所有文件
for dir in os.listdir(folder_path):
    if not os.path.isdir(os.path.join(folder_path, dir)):
        continue
    for filename in os.listdir(dir):
        # 判断是否为 txt 文件
        if filename.endswith(".txt"):
            # 打开文件并读取所有行
            with open(os.path.join(folder_path, dir, filename), "r") as f:
                lines = f.readlines()
            
            estimated = ""
            begin = ""
            end = ""
            for line in lines:
                if "Train Epoch: 1(0%)" in line and "Experiment Finished Time" in line:
                    estimated = line
                if "Test Epoch: 0" in line:
                    begin = line
                if "Test Epoch: 20" in line:
                    end = line
            
            print("====", dir, filename, "====")
            if begin == "" or end == "":
                print(f"未找到足够的时间信息")
                continue
            time1 = datetime.strptime(begin[:19], "%Y-%m-%d %H:%M:%S")
            time2 = datetime.strptime(end[4:23], "%Y-%m-%d %H:%M:%S")
            diff = time2 - time1
            print(f"实际时间差 {diff}, 预计时间差 {estimated[-16:-8]}")