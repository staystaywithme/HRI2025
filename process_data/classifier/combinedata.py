import numpy as np
import os
import pandas as pd

def process_data(data):
    data_length = len(data)

    if data_length < 301:
        # 上采样（插值）
        new_indices = np.linspace(0, data_length - 1, 301)
        data = pd.DataFrame(data).reindex(new_indices).interpolate(method='linear').values.flatten()
    elif data_length > 301:
        # 下采样
        indices = np.round(np.linspace(0, data_length - 1, 301)).astype(int)
        data = data[indices]

    return data

AC = np.empty((0, 301, 12))
AD = np.empty((0, 301, 12))
BC = np.empty((0, 301, 12))
BD = np.empty((0, 301, 12))

def combine_data(name, data_type):
    global AC, AD, BC, BD
    for i in range(1, 11):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/classifier/{name}_{data_type}{i:02}_classify.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',')
        print(data.shape)
        data = process_data(data)
        print(data.shape)
        data = np.expand_dims(data, axis=0)
        if data_type == "AC":
            AC = np.concatenate((AC, data))
        elif data_type == "AD":
            AD = np.concatenate((AD, data))
        elif data_type == "BC":
            BC = np.concatenate((BC, data))
        elif data_type == "BD":
            BD = np.concatenate((BD, data))
    print(AC.shape)
    print(AD.shape)
    print(BC.shape)
    print(BD.shape)

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]

# 调用函数处理每个名称和类型组合
for name in names:
    for data_type in types:
        combine_data(name, data_type)