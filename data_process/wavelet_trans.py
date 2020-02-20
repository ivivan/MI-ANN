import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd





# datafile
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-multiple.csv'

dataset = pd.read_csv(filepath)
dataset = dataset.iloc[0:0 + 50, :]
dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)


print(dataset.head())

ntu_list = dataset['Turbidity_NTU'].tolist()


# 函数打包

def wt(index_list, wavefunc, lv, m,
       n):  # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数

    # 分解
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=lv)  # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn函数

    # 去噪过程
    for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log2(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零

    # 重构
    denoised_index = pywt.waverec(coeff, wavefunc)
    return denoised_index






# 调用函数wt
ts_rec = wt(ntu_list, 'db4', 2, 1, 2)  # 小波函数为db4, 分解层数为4， 选出小波层数为1-4层


def addwavelettoorigin(wavelet_list,dataframe,columnname):
    temp_series = pd.Series(wavelet_list)
    dataframe[columnname]= temp_series.values


addwavelettoorigin(ts_rec,dataset,'Turbidity_wavelet')

print(dataset.head())




