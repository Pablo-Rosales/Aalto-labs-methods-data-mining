from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial import distance
import statistics 

#The data set 1 declared
data = np.array([[0,-1/2,3/2,1], [1,3/2,5/2,3]])

print(data)
#Plotting the dataset
#Exctract two vectors (one for x, the other one for y)
array_x = data[0]
array_y = data[1]

plt.plot(array_x, array_y, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset 1")
plt.show()

#The data set 2 is declared
data2 = np.array([[math.sqrt(0.5), math.sqrt(0.5), 4*math.sqrt(0.5), 4*math.sqrt(0.5)],[math.sqrt(0.5), 2*math.sqrt(0.5), math.sqrt(0.5), 2*math.sqrt(0.5)]])

array_x2 = data2[0]
array_y2 = data2[1]


plt.plot(array_x2, array_y2, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset 2")
plt.show()

fig, (dataset1,dataset2) = plt.subplots(1,2, figsize = (20,10))
fig.suptitle("Dataset 1 - Dataset 2")
dataset1.plot(array_x, array_y,"o")
dataset2.plot(array_x2, array_y2,"o")
