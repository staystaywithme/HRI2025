import numpy as np
import os

def select_data(name, type):
    for i in range(1, 11):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/gomi/reduce0/{name}_{type}{i:02}_processed.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        # 加载数据
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        # 过滤掉第19列中等于1.0的行
        filtered_data = data[data[:, 18] != 1.0]
        # 找出从1变成不是1的时间点
        change_from_1 = np.where((data[:-1, 18] == 1.0) & (data[1:, 18] != 1.0))[0] + 1
        # 找出从不是1变成1的时间点
        change_to_1 = np.where((data[:-1, 18] != 1.0) & (data[1:, 18] == 1.0))[0] + 1
        # 提取从1变成不是1的时间点的前20行
        pre_change_from_1 = np.concatenate([data[max(0, idx-50):idx] for idx in change_from_1])
        # 提取从不是1变成1的时间点的后20行
        post_change_to_1 = np.concatenate([data[idx:min(len(data), idx+50)] for idx in change_to_1])
        # 合并结果
        result = np.concatenate((filtered_data, pre_change_from_1, post_change_to_1))

        data = np.hstack((result[:, 3:8], result[:, 12:17]))
        output_dir = "/Users/syunsei/Desktop/SII2025/process_data/classifier"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{name}_{type}{i:02}_classify.csv")
        np.savetxt(output_file_path, data, delimiter=',', fmt='%f')
        print(f"Processed file saved to {output_file_path}")
    return data

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]

# 调用函数处理每个名称和类型组合
for name in names:
    for type in types:
        select_data(name, type)