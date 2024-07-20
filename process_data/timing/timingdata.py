import numpy as np
import os
import matplotlib.pyplot as plt

start_train = np.empty((0, 200, 12))
end_train = np.empty((0, 200, 12))
start_val = np.empty((0, 200, 12))
end_val = np.empty((0, 200, 12))
start_test = np.empty((0, 200, 12))
end_test = np.empty((0, 200, 12))

#selectdata
def select_train_data(name, type):
    for i in range(1,7):
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
        # 提取从1变成不是1的时间点的前_行
        pre_change_from_1_1 = np.concatenate([data[max(0, idx-100):idx] for idx in change_from_1])
        pre_change_from_1_2 = np.concatenate([data[max(0, idx+100):idx] for idx in change_from_1])
        # 提取从不是1变成1的时间点的后_行
        post_change_to_1_1 = np.concatenate([data[idx:min(len(data), idx+100)] for idx in change_to_1])
        post_change_to_1_2 = np.concatenate([data[idx:min(len(data), idx-100)] for idx in change_to_1])
        start = np.concatenate(pre_change_from_1_1, pre_change_from_1_2)
        end = np.concatenate(post_change_to_1_1, post_change_to_1_2)
        start =np.hstack(start[:,3:9], end[:,12:18])
        end = np.hstack(start[:,3:9], end[:,12:18])
        #data = np.hstack((result[:, 3:9], result[:, 12:18]))

        start_train = np.concatenate((start_train, start))
        end_train = np.concatenate((end_train, end))
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/start_train.npy", start_train)
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/end_train.npy", end_train)
        '''if name in lists:
            data = data * -1
        
        if name == "zhou" and type == "BC" and i == 1:
            data = data * -1'''
    return data




def select_val_data(name, type):
    for i in range(7, 9):
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
        # 提取从1变成不是1的时间点的前_行
        pre_change_from_1_1 = np.concatenate([data[max(0, idx-100):idx] for idx in change_from_1])
        pre_change_from_1_2 = np.concatenate([data[max(0, idx+100):idx] for idx in change_from_1])
        # 提取从不是1变成1的时间点的后_行
        post_change_to_1_1 = np.concatenate([data[idx:min(len(data), idx+100)] for idx in change_to_1])
        post_change_to_1_2 = np.concatenate([data[idx:min(len(data), idx-100)] for idx in change_to_1])
        start = np.concatenate(pre_change_from_1_1, pre_change_from_1_2)
        end = np.concatenate(post_change_to_1_1, post_change_to_1_2)
        start =np.hstack(start[:,3:9], end[:,12:18])
        end = np.hstack(start[:,3:9], end[:,12:18])
        #data = np.hstack((result[:, 3:9], result
        start_val = np.concatenate((start_val, start))
        end_val = np.concatenate((end_val, end))
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/start_val.npy", start_val)
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/end_val.npy", end_val)


def select_test_data(name, type):
    for i in range(9, 11):
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
        # 提取从1变成不是1的时间点的前_行
        pre_change_from_1_1 = np.concatenate([data[max(0, idx-100):idx] for idx in change_from_1])
        pre_change_from_1_2 = np.concatenate([data[max(0, idx+100):idx] for idx in change_from_1])
        # 提取从不是1变成1的时间点的后_行
        post_change_to_1_1 = np.concatenate([data[idx:min(len(data), idx+100)] for idx in change_to_1])
        post_change_to_1_2 = np.concatenate([data[idx:min(len(data), idx-100)] for idx in change_to_1])
        start = np.concatenate(pre_change_from_1_1, pre_change_from_1_2)
        end = np.concatenate(post_change_to_1_1, post_change_to_1_2)
        start =np.hstack(start[:,3:9], end[:,12:18])
        end = np.hstack(start[:,3:9], end[:,12:18])
        #data = np.hstack((result[:, 3:9], result
        start_test = np.concatenate((start_test, start))
        end_test = np.concatenate((end_test, end))
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/start_test.npy", start_test)
        np.save(f"/Users/syunsei/Desktop/SII2025/process_data/timing/end_test.npy", end_test)
        



names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]
lists = [ "liu", "wang", "zhou"]

# 调用函数处理每个名称和类型组合
for name in names:
    for type in types:
        select_train_data(name, type)
        select_val_data(name, type)
        select_test_data(name, type)
        