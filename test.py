# import other modules needed, install them if cannot found
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

datPath = r"C:\Users\ZHA244\Downloads\sp500_short_period.csv"

# read data from csv file
with open(datPath) as csvfile:
    csvData = csv.reader(csvfile)
    datList = []
    for row in csvData:
        datList.append(row)

# get the colnames in the first row and remove it
symbols = np.array(datList.pop(0))

# convert list to matrix
data = np.array(datList)
data = data.astype(np.float)

print('here')

# PCA
k = 3  # number of principal components
n = np.size(data, 1)
covm = np.cov(data, rowvar=False)
eig_val, eig_vec = np.linalg.eig(covm)
eig_val = np.sort(eig_val)[::-1]  # sort descending
loc = eig_val.argsort()[::-1]  # Use [::-1] to reverse a list, can also use reverse()
E = eig_vec[:, loc]
w = E[:, 0:k]
l2norm = np.sqrt(np.sum(w ** 2, axis=0))
w = np.divide(w, np.dot(np.ones(n)[:, None], l2norm.T[None, :]))  # .T to transpose an array
datapca = np.dot(data, w)

# Or can compute PCA using scikit-learn, however can not get the eigenvectors
# pca = PCA(n_components=3)
# datapca = pca.fit_transform(data)  # Reconstruct signals based on orthogonal components

print('heretwo')

# Compute ICA
ica = FastICA(n_components=k)
S = ica.fit_transform(datapca)  # Reconstruct signals
print('herefour')

A = ica.mixing_  # Get estimated mixing matrix
rec_data = np.dot(np.dot(S, A.T) + ica.mean_, w.T)

plotvars = 6  # number of variables to plot, must be < n

print('three')
for i in range(plotvars):
    plt.figure()
    plt.plot(data[i, :], "r", label="Original price")
    plt.plot(rec_data[i, :], 'b', label="Recovered price")
    plt.title("Original and reconstructed stock prices for symbol " + symbols[i])
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()

plt.show()