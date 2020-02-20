from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression,chi2,variance_threshold,SelectFromModel,SelectPercentile
from sklearn.linear_model import Lasso
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler





# data input
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv'

# filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\all_season.csv'

dataset = pd.read_csv(filepath) #384 5 days
#
# dataset = dataset.fillna(method='pad')
# print("Display rows contain NaNs again--------")
# print(dataset[dataset.isnull().T.any().T])

# dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'],dayfirst=True)
# dataset['TimeNumber'] = dataset['TIMESTAMP'].apply(lambda x: x.timestamp()).values
#
# Y = dataset['DO_mg'].values



dataset = dataset.drop(dataset.columns[0],axis=1)
dataset = dataset.drop(['TIMESTAMP','DO_Sat'],axis=1)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(dataset)
dataset.loc[:,:] = scaled_values

X = dataset.as_matrix()
Y = dataset['DO_mg'].values

pd.set_option('display.max_columns', None)
print(dataset.head())



# Pearsonr

for feature in range(6):
    print(sc.stats.pearsonr(dataset.iloc[:,feature].values,Y))


# print(sc.stats.pearsonr(a,b))

# Select12=SelectPercentile(chi2,precentile=30)
# X12=Select12.fit_transform(X,Y)
# print(Select12.scores_)
#
# # 方差
#
# Select2 = Lasso(alpha=0.1).fit(X, Y)
# model = SelectFromModel(Select2, prefit=True)
# X_new = model.transform(X)
# print(X_new.shape)
# print(X_new)

# # 互信息
# Select3=SelectKBest(mutual_info_regression,k=3)
# X3=Select3.fit_transform(X,Y)
# print(Select3.scores_)
# print(X3)

# # f_regression
# Select4=SelectKBest(f_regression,k=3)
# X4=Select4.fit_transform(X,Y)
# print(Select4.scores_)





#
#
# f_test, _ = f_regression(X, y)
# f_test /= np.max(f_test)
#
# mi = mutual_info_regression(X, y)
# mi /= np.max(mi)
#
# plt.figure(figsize=(15, 5))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.scatter(X[:, i], y, edgecolor='black', s=20)
#     plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
#     if i == 0:
#         plt.ylabel("$y$", fontsize=14)
#     plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
#               fontsize=16)
# plt.show()