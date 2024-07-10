import numpy as np
import os
import pandas as pd

def wash_data(name, type):
    data = []
    for i in range(1,11):
        file_path = f"\\Users\\syunsei\\Desktop\\SII2025\\process_data\\ごみ\\{name}_{type}{i:02}_washed.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        data = data[data.iloc[:, 4] != -0.0625]
        output_dir = "\\Users\\syunsei\\Desktop\\SII2025\\process_data\\ごみ"
        output_file_path = os.path.join(output_dir, f"{name}_{type}{i:02}_washed.csv")
        np.savetxt(output_file_path, data, delimiter=',', fmt='%f')
    return data

names = ["gashi", "jingchen", "liu", "qing", "wang", "zhou"]
types = ["AC", "AD", "BC", "BD"]

# Call the function for each combination of name and type
for name in names:
    for type in types:
        wash_data(name, type)
