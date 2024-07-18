import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def reduce0(data):
    # 将 NumPy 数组转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    # 过滤第2,8列值为 -0.0625 的行
    df = df[df.iloc[:, 1] != -0.0625]
    df = df[df.iloc[:, 7] != -0.0625]
    data = df.to_numpy()
    return data

def reducenoice(data):
    # 对数据进行平滑处理
    # 对每一列进行平滑处理（假设列数为n）
    for col in range(data.shape[1]):
        data[:, col] = savgol_filter(data[:, col], window_length=11, polyorder=10)
    return data

def data301(data):
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