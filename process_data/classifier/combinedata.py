import numpy as np
import os
import matplotlib.pyplot as plt

def process_data(data):
    data_length = len(data)

    if data_length < 301:
        # 上采样（插值）
        step, features = data.shape
        new_length = 301
        old_length = step
        new_time = np.linspace(0, 1, new_length)
        old_time = np.linspace(0, 1, old_length)
        unsampled = np.zeros((new_length, features))
        for j in range(features):
            unsampled[:, j] = np.interp(new_time, old_time, data[:, j])
        data = unsampled
    elif data_length > 301:
        # 下采样
        indices = np.round(np.linspace(0, data_length - 1, 301)).astype(int)
        data = data[indices]

    return data

#plot
labels = ["xGyro1", "yGyro1", "zGyro1", "xAccl1", "yAccl1", "zAccl1", 
          "Roll1", "Pitch1", "Yaw1", "xGyro2", "yGyro2", "zGyro2", 
          "xAccl2", "yAccl2", "zAccl2", "Roll2", "Pitch2", "Yaw2"]
def plot_segments(data_segments):
    for i, segment in enumerate(data_segments):
        plt.figure(figsize=(14, 7))  # 设置图片大小
        for feature_index in range(segment.shape[1]):  # 假设每个段都有18个特征
            plt.plot(segment[:,feature_index], label=labels[feature_index])
        plt.title(f'{data_type} {name} Segment {i}')
        i += 1 
        plt.xlabel('Time Point', fontsize=20)
        plt.ylabel('Value', fontsize=20)
        plt.xticks(fontsize=20)  # Increase font size of x-tick labels
        plt.yticks(fontsize=20) 
        plt.legend(loc='upper right', fontsize=12)  # 可能需要调整位置或去除，如果图例太大或重叠
        plt.show()

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
    np.save(f'/Users/syunsei/Desktop/SII2025/process_data/classifier/{data_type}.npy', eval(data_type))

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]

# 调用函数处理每个名称和类型组合
for name in names:
    for data_type in types:
        combine_data(name, data_type)
        #plot_segments(AC)
