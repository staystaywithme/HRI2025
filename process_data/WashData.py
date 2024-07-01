import numpy as np
import os

def wash_data(name, type):
    data = []
    for i in range(1,11):
        file_path = f"C:\\Github_LIU\\SII2025\\Raw_Data\\{name}\\{name}_{type}{i:02}.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        data = data[~(data > 1000).any(axis=1)]
        output_dir = "C:\\Github_LIU\\SII2025\\process_data"
        output_file_path = os.path.join(output_dir, f"{name}_{type}{i:02}_washed.csv")
        np.savetxt(output_file_path, data, delimiter=',', fmt='%f')
    return data

wash_data("gashi","AC")
wash_data("gashi","AD")
wash_data("gashi","BC")
wash_data("gashi","BD")

wash_data("jingchen","AC")
wash_data("jingchen","AD")
wash_data("jingchen","BC")
wash_data("jingchen","BD")

wash_data("liu","AC")
wash_data("liu","AD")
wash_data("liu","BC")
wash_data("liu","BD")

wash_data("qing","AC")
wash_data("qing","AD")
wash_data("qing","BC")
wash_data("qing","BD")

wash_data("wang","AC")
wash_data("wang","AD")
wash_data("wang","BC")
wash_data("wang","BD")

wash_data("zhou","AC")
wash_data("zhou","AD")
wash_data("zhou","BC")
wash_data("zhou","BD")