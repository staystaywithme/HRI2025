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

AC_train = np.empty((0, 301, 12))
AD_train = np.empty((0, 301, 12))
BC_train = np.empty((0, 301, 12))
BD_train = np.empty((0, 301, 12))

AC_val = np.empty((0, 301, 12))
AD_val = np.empty((0, 301, 12))
BC_val = np.empty((0, 301, 12))
BD_val = np.empty((0, 301, 12))

AC_test = np.empty((0, 301, 12))
AD_test = np.empty((0, 301, 12))
BC_test = np.empty((0, 301, 12))
BD_test = np.empty((0, 301, 12))

def combine_train_data(name, data_type,timing):
    global AC_train, AD_train, BC_train, BD_train
    for i in range(1, 7):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/classifier/{name}_{data_type}{i:02}_classify_{timing}.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',')
        print(data.shape)
        data = process_data(data)
        print(data.shape)
        data = np.expand_dims(data, axis=0)
        if data_type == "AC":
            AC_train = np.concatenate((AC_train, data))
        elif data_type == "AD":
            AD_train = np.concatenate((AD_train, data))
        elif data_type == "BC":
            BC_train = np.concatenate((BC_train, data))
        elif data_type == "BD":
            BD_train = np.concatenate((BD_train, data))
    print(AC_train.shape)
    print(AD_train.shape)
    print(BC_train.shape)
    print(BD_train.shape)
    np.save(f'/Users/syunsei/Desktop/SII2025/process_data/classifier/{data_type}_train.npy', eval(f'{data_type}_train'))

def combine_val_data(name, data_type,timing):
    global AC_val, AD_val, BC_val, BD_val
    for i in range(7, 9):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/classifier/{name}_{data_type}{i:02}_classify_{timing}.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',')
        data = process_data(data)
        data = np.expand_dims(data, axis=0)
        if data_type == "AC":
            AC_val = np.concatenate((AC_val, data))
        elif data_type == "AD":
            AD_val = np.concatenate((AD_val, data))
        elif data_type == "BC":
            BC_val = np.concatenate((BC_val, data))
        elif data_type == "BD":
            BD_val = np.concatenate((BD_val, data))
    print(AC_val.shape)
    print(AD_val.shape)
    print(BC_val.shape)
    print(BD_val.shape)
    np.save(f'/Users/syunsei/Desktop/SII2025/process_data/classifier/{data_type}_val.npy', eval(f'{data_type}_val'))

def combine_test_data(name, data_type,timing):
    global AC_test, AD_test, BC_test, BD_test
    for i in range(9, 11):
        file_path = f"/Users/syunsei/Desktop/SII2025/process_data/classifier/{name}_{data_type}{i:02}_classify_{timing}.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',')
        data = process_data(data)
        data = np.expand_dims(data, axis=0)
        if data_type == "AC":
            AC_test = np.concatenate((AC_test, data))
        elif data_type == "AD":
            AD_test = np.concatenate((AD_test, data))
        elif data_type == "BC":
            BC_test = np.concatenate((BC_test, data))
        elif data_type == "BD":
            BD_test = np.concatenate((BD_test, data))
    print(AC_test.shape)
    print(AD_test.shape)
    print(BC_test.shape)
    print(BD_test.shape)
    np.save(f'/Users/syunsei/Desktop/SII2025/process_data/classifier/{data_type}_test.npy', eval(f'{data_type}_test'))

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]
timings = {"80", "90","100", "110", "120"}

# 调用函数处理每个名称和类型组合
for name in names:
    for data_type in types:
        for timing in timings:
            combine_train_data(name, data_type,timing)
            combine_val_data(name, data_type,timing)
            combine_test_data(name, data_type,timing)
            #plot_segments(AC)
