import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter

def wash_data(name, type):
    for i in range(1, 11):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/gomi/reduce0/{name}_{type}{i:02}_processed.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        # 加载数据
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        # 将 NumPy 数组转换为 Pandas DataFrame
        df = pd.DataFrame(data)
        # 过滤第五列值为 -0.0625 的行
        df = df[df.iloc[:, 4] != -0.0625]
        df = df[df.iloc[:, 13] != -0.0625]
        
        # 对数据进行平滑处理
        # 对每一列进行平滑处理（假设列数为n）
        for col in range(df.shape[1]):
            df.iloc[:, col] = savgol_filter(df.iloc[:, col], window_length=11, polyorder=10)
        
        # 转换回 NumPy 数组保存
        data = df.to_numpy()
        output_dir = "/Users/syunsei/Desktop/SII2025/process_data/gomi/reduce0"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{name}_{type}{i:02}_processed.csv")
        np.savetxt(output_file_path, data, delimiter=',', fmt='%f')
        print(f"Processed file saved to {output_file_path}")
    return data

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]

# 调用函数处理每个名称和类型组合
for name in names:
    for type in types:
        wash_data(name, type)