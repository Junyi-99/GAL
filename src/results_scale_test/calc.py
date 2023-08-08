# calculator for scale test
import os
import re
from datetime import datetime
from collections import defaultdict
# 定义文件夹路径
folder_path = "./"

# 遍历文件夹中的所有文件
final_result = defaultdict(list)
for dir in sorted(os.listdir(folder_path)):
    if not os.path.isdir(os.path.join(folder_path, dir)):
        continue
    for filename in sorted(os.listdir(dir)):
        # 判断是否为 txt 文件
        if not filename.endswith(".txt"):
            continue

        # 打开文件并读取所有行
        with open(os.path.join(folder_path, dir, filename), "r") as f:
            lines = f.readlines()
        
        current_progress = ""
        result = ""
        for line in lines:
            if "Test Epoch: 20" in line:
                result = line
            if "Test Epoch: " in line:
                current_progress = line

        result = result[result.find("Test Epoch: 20") + 14:-1]
        method = dir.split("_")[0]
        client = dir.split("_")[1]
        seed   = dir.split("_")[3]

        if result == "":
            progress = current_progress.split(' ')[7]
            pos = progress.find("(")
            progress = int(progress[:pos])
            print(f"🔴 GAL 没跑完 {method}, {client}, {seed}, Progress: {progress}/20 ({round(progress/20 * 100,0)}%)")
            continue
        else:
            print(f"🟢 GAL 跑完了 {method}, {client}, {seed}, {result}")
        
        result = result[result.find("Accuracy: ") + 10: ]
        final_result[method+"_"+client].append(float(result))
        

for key, value in final_result.items():
    print(f"{key}: {round(sum(value)/len(value), 4)}")