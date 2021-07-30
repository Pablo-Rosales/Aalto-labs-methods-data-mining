# Pablo Rosales Rodríguez, Student id: 914769
# Assignment 2

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial import distance
import statistics 

#EXERCISE 1
df = pd.read_csv("spiral.txt", sep="\t", header=None)

plt.figure(0)
plt.rcParams["figure.figsize"] = (15,5)
plt.style.use('ggplot')
plt.clf()
plt.title('Features and Ground-truth labels')
plt.plot(df,'.')
plt.legend(['Data feature 0','Data feature 1','Ground-truth labels'])

X = np.array(df.drop(columns = 2))


plt.figure(num=1, figsize=(3,3))
plt.clf()
plt.title("Ground-truth clusters")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
for i in range(len(X)):
    if df[2][i] == 1:
        plt.plot(df[0][i], df[1][i], 'o', color='grey')
    elif df[2][i] == 2:
        plt.plot(df[0][i], df[1][i], 'o', color='blue')
    else:
        plt.plot(df[0][i], df[1][i], 'o', color='red')

#CLUSTERING
K = 3

#k-MEANS CLUSTERING
print('\nk-means:')
kmeans = KMeans(n_clusters=K).fit(X)
labels = kmeans.labels_

#Goodness of clustering using different metrics
#Silhouette Coefficient
silhouette_result = metrics.silhouette_score(X, labels, metric='euclidean')
print("\nSilhouette Coefficient")
print(silhouette_result)
#Davies-Bouldin
print("\nDavies-Bouldin Score:")
bouldin_result = metrics.davies_bouldin_score(X, labels)
print(bouldin_result)
#Normalized Mutual Information
print("\nNormalized Mutual Information:")
mutual_information = metrics.normalized_mutual_info_score(df[2], labels)
print(mutual_information)
print("\n")

plt.figure(num=2, figsize = (3,3))
plt.clf()
plt.title("k-means clusters")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
for i in range(len(X)):
    if labels[i] == 0:
        plt.plot(df[0][i], df[1][i], 'o', color='red')
    elif labels[i] == 1:
        plt.plot(df[0][i], df[1][i], 'o', color='grey')
    else:
        plt.plot(df[0][i], df[1][i], 'o', color='blue')

#SPECTRAL CLUSTERING
print('\nSpectral clustering:')
kernel_gamma = 1
spectral = SpectralClustering(n_clusters=K, gamma=kernel_gamma).fit(X)
spectral_labels = spectral.labels_

#Goodness of clustering using different metrics
#Silhouette Coefficient
silhouette_result = metrics.silhouette_score(X, spectral_labels, metric='euclidean')
print("\nSilhouette Coefficient")
print(silhouette_result)
#Davies-Bouldin
print("\nDavies-Bouldin Score:")
bouldin_result = metrics.davies_bouldin_score(X, spectral_labels)
print(bouldin_result)
#Normalized Mutual Information
print("\nNormalized Mutual Information:")
mutual_information = metrics.normalized_mutual_info_score(df[2], spectral_labels)
print(mutual_information)
print("\n")

plt.figure(num=3, figsize=(3,3))
plt.clf()
plt.title("Spectral clusters")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
for i in range(len(X)):
    if spectral_labels[i] == 0:
        plt.plot(df[0][i], df[1][i], 'o', color='blue')
    elif spectral_labels[i] == 1:
        plt.plot(df[0][i], df[1][i], 'o', color='grey')
    else:
        plt.plot(df[0][i], df[1][i], 'o', color='red')


#EXERCISE 2
#Kernel Matrix
kernel_matrix = np.zeros([len(X),len(X)])
sigma = 0.5
for i in range(len(X)):
    for j in range(len(X)):
        kernel_matrix[i,j] = np.exp((-np.linalg.norm(X[i]-X[j])**2)/(2*sigma**2))

#Thao function
def thao(kernel_matrix, cluster_labels):
    
    sum_total = 0
    n = len(cluster_labels)
    
    for i in range(n):
        den = 0
        num = 0
        for j in range(n):
            if i != j:
                if cluster_labels[i] == cluster_labels[j]:
                    c = 1
                else:
                    c = 0
                    
                num = c * kernel_matrix[i,j] + num;
                den = kernel_matrix[i,j] + den;
                    
        sum_total = num / den + sum_total
            
    thao = 1/n * sum_total
    
    return thao

thao_kmeans = thao(kernel_matrix, labels)
thao_spectral = thao(kernel_matrix, spectral_labels)
print("\nThao kmeans")
print(thao_kmeans)
print("\nThao spectral")
print(thao_spectral)

#Alternative validation index
euclidean_matrix = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        euclidean_dist = math.sqrt((X[j][0]-X[i][0])**2 + (X[j][1]-X[i][1])**2)
        euclidean_matrix[i][j] = euclidean_dist

alternative_index_kmeans = thao(euclidean_matrix, labels)
alternative_index_spectral = thao(euclidean_matrix, spectral_labels)

print("\nAlternative Index Kmeans")
print(alternative_index_kmeans)
print("\nAlternative index spectral")
print(alternative_index_spectral)


#EXERCISE 3
patients = 1000
p_hd = 0.3

#Compute probabilities from frequencies
frx = np.array([300,500,500,342,2,500,260,120,240,80,200,251])
frxc = np.array([125,150,400,240,2,352,100,32,100,32,100,203])

Px = np.array(frx/patients)
Pc = np.array([p_hd,p_hd,1-p_hd,1-p_hd,1-p_hd,1-p_hd,p_hd,p_hd,p_hd,p_hd,p_hd,1-p_hd])
Pxc = np.array(frxc/patients)

leverage = np.array(Pxc-Px*Pc)
print("\n")
print("\nLEVERAGE WITHOUT PRUNING:")
print(leverage)

lift = np.array(Pxc/(Px*Pc))
print("\nLIFT WITHOUT PRUNING:")
print(lift)

#Those rules with leverage >0 and lift >1 are positively associated
index = []
for i in range(len(leverage)):
    if (leverage[i]<=0 and lift[i]<=1):
        index.append(i)
        

frx = np.delete(frx, index)
frxc = np.delete(frxc, index)
Px = np.delete(Px, index)
Pc = np.delete(Pc, index)
Pxc = np.delete(Pxc, index)
leverage = np.delete(leverage, index)
lift = np.delete(lift, index)       
        
print("\n")
print("\nfrx WITH PRUNING")
print(frx)
print("\nLEVERAGE WITH PRUNING:")
print(leverage)

print("\nLIFT WITH PRUNING:")
print(lift)

#Compute mutual information
MI = np.zeros(len(leverage))
for i in range(len(leverage)):
    num1 = Pxc[i]
    num2 = (frx[i]*(1-Pc[i])-patients*leverage[i])/patients
    num3 = ((patients-frx[i])*Pc[i]-patients*leverage[i])/patients
    num4 = ((patients-frx[i])*(1-Pc[i])+patients*leverage[i])/patients
    
    den1 = Px[i]
    den2 = 1-Px[i]
    den3 = Pc[i]
    den4 = 1-Pc[i]
    
    num = (num1**num1 * num2**num2 * num3**num3 * num4**num4)
    den = (den1**den1 * den2**den2 * den3**den3 * den4**den4)
    
    MI[i] = patients * np.log2(num/den)

print("\nMutual Information")
print(MI)
index = []
for i in range(len(leverage)):
    if (MI[i] < 1.5):
        index.append(i)

frx = np.delete(frx, index)
frxc = np.delete(frxc, index)
Px = np.delete(Px, index)
Pc = np.delete(Pc, index)
Pxc = np.delete(Pxc, index)
leverage = np.delete(leverage, index)
lift = np.delete(lift, index)  













