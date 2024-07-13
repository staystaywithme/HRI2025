import numpy as np
import os
import matplotlib.pyplot as plt

#plot
labels = ["xGyro1", "yGyro1", "zGyro1", "xAccl1", "yAccl1", "zAccl1", "xGyro2", "yGyro2", "zGyro2",  "xAccl2", "yAccl2", "zAccl2", ]
def plot_sample(data, name, type, file_index):
    for i, segment in enumerate(data):
        plt.figure(figsize=(14, 7))  # 设置图片大小
        plt.plot(segment[:,0:3], label=labels[0:3])
        plt.title(f'{type} {name} File {file_index}')
        plt.xlabel('Time Point', fontsize=20)
        plt.ylabel('Value', fontsize=20)
        plt.xticks(fontsize=20)  # Increase font size of x-tick labels
        plt.yticks(fontsize=20) 
        plt.legend(loc='upper right', fontsize=12)  # 可能需要调整位置或去除，如果图例太大或重叠
        plt.show()

#selectdata
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
    
        data = np.hstack((result[:, 3:9], result[:, 12:18]))
        
        if name in lists:
            data = data * -1
        
        if name == "zhou" and type == "BC" and i == 1:
            data = data * -1
        
        output_dir = "/Users/syunsei/Desktop/SII2025/process_data/classifier"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{name}_{type}{i:02}_classify.csv")
        np.savetxt(output_file_path, data, delimiter=',', fmt='%f')
        print(f"Processed file saved to {output_file_path}")

        #plot_sample([data],name, type, i)
    return data

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]
lists = [ "liu", "wang", "zhou"]

# 调用函数处理每个名称和类型组合
for name in names:
    for type in types:
        select_data(name, type)
        