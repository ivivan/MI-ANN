import pandas as pd
import numpy as np
from statsmodels.multivariate.pca import PCA
from matplotlib import pyplot as plt





# data input
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv'

dataset = pd.read_csv(filepath)
# dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'],dayfirst=True)
# dataset['TimeNumber'] = dataset['TIMESTAMP'].apply(lambda x: x.timestamp()).values

Y = dataset['DO_mg'].values




dataset = dataset.drop(dataset.columns[0],axis=1)
dataset = dataset.drop(['TIMESTAMP','DO_Sat','DO_mg'],axis=1)

# X = dataset.as_matrix()

print(dataset.head())
print(dataset.shape)


pc = PCA(dataset, ncomp=2,method='svd')

print(pc.factors.shape)

print(pc.factors)