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

#EXERCISE 1
# The data is loaded from the csv file
df = pd.read_csv("nba2013_data.csv")

#Those columns containing non-numerical features are pruned
df = df.drop(columns = ['player', 'pos', 'bref_team_id', 'season'])
#Those values equal to NA are replaced with the mean of their column
df.fillna(df.mean(), inplace=True)
#The data is transformed into an array
X = np.array(df)

#k-means clustering using KMeans from sklearn library with k number of clusters
k = 5;
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#Goodness of clustering using different metrics
#Silhouette Coefficient

silhouette_result = metrics.silhouette_score(X, labels, metric='euclidean')
print("\nSilhouette Coefficient")
print(silhouette_result)
#Calinski Harabasz
print("\nCalinski-Harabasz Score:")
calinski_result = metrics.calinski_harabasz_score(X, labels)
print(calinski_result)
#Davies-Bouldin
print("\nDavies-Bouldin Score:")
bouldin_result = metrics.davies_bouldin_score(X, labels)
print(bouldin_result)
print("\n")

print("-----------------------------------------------------------------------------------")
#Optimal k
#By using one of the previous computed metrics, such as the Silhouette coefficient
##########################################################################################
max_silhouette = 0
k_optimal_silhouette = 0
x_silhouette = []
y_silhouette = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette_result_2 = metrics.silhouette_score(X, labels, metric='euclidean')
    x_silhouette.append(i)
    y_silhouette.append(silhouette_result_2)
    if silhouette_result_2 > max_silhouette:
        max_silhouette = silhouette_result_2
        k_optimal_silhouette = i

#print(k_optimal_silhouette)
#plt.plot(x_silhouette, y_silhouette, 'cyan')
#plt.show()

#Using Calinski Harabasz
max_calinski = 0
k_optimal_calinski = 0
x_calinski = []
y_calinski = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    labels = kmeans.labels_
    calinski_result_2 = metrics.calinski_harabasz_score(X, labels)
    x_calinski.append(i)
    y_calinski.append(calinski_result_2)
    if calinski_result_2 > max_calinski:
        max_calinski = calinski_result_2
        k_optimal_calinski = i

#print(k_optimal_calinski)
#plt.plot(x_calinski, y_calinski, 'red')
#plt.show()

#Using Davies-Bouldin 
max_bouldin = 0
k_optimal_bouldin = 0
x_bouldin = []
y_bouldin = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    labels = kmeans.labels_
    bouldin_result_2 = metrics.davies_bouldin_score(X, labels)
    x_bouldin.append(i)
    y_bouldin.append(bouldin_result_2)
    if bouldin_result_2 > max_bouldin:
        max_bouldin = bouldin_result_2
        k_optimal_bouldin = i

#print(k_optimal_bouldin)
#plt.plot(x_bouldin, y_bouldin, 'green')
#plt.show()

fig, (silhouette,calinski,davies) = plt.subplots(1,3, figsize = (20,10))
fig.suptitle("Silhouette Coefficient - Calinski Harabasz Score - Davies-Bouldin Score")
silhouette.plot(x_silhouette, y_silhouette, 'cyan')
calinski.plot(x_calinski, y_calinski, 'red')
davies.plot(x_bouldin, y_bouldin, 'green')


##########################################################################################
print("-----------------------------------------------------------------------------------")
#EXERCISE 2
#Hierarchical agglomerative clustering using sklearn library
#Single-linkage metric
clustering_single = AgglomerativeClustering(n_clusters=k, linkage="single").fit(X)
single_labels = clustering_single.labels_
#Complete-linkage metric
clustering_complete = AgglomerativeClustering(n_clusters=k, linkage="complete").fit(X)
complete_labels = clustering_complete.labels_
#Average-linkage metric
clustering_average = AgglomerativeClustering(n_clusters=k, linkage="average").fit(X)
average_labels = clustering_average.labels_
#Distance of centroids metric
#Array containing an upper triangular of the distance matrix
Y = sp.spatial.distance.pdist(X, metric = "euclidean")
#Performs the clustering with the "distance of centroids" metric
Z = sp.cluster.hierarchy.linkage(Y, method = "centroid")
#Transformation in flat clusters from the previous hierarchical clustering
Z = sp.cluster.hierarchy.fcluster(Z, k, criterion = "maxclust")

#Optimal Metric Evaluation
#Considering the Silhouette Coefficient
max_silhouette = 0
k_optimal_silhouette = 0
x_silhouette = []
y_silhouette = []
silhouette_result_single = 0
silhouette_result_complete = 0
silhouette_result_average = 0
silhouette_result_centroid = 0
y_single_silhouette =[]
y_complete_silhouette = []
y_average_silhouette = []
y_centroid_silhouette = []




for i in range(2, 20):
    clustering_single = AgglomerativeClustering(n_clusters=i, linkage="single").fit(X)
    single_labels = clustering_single.labels_
    clustering_complete = AgglomerativeClustering(n_clusters=i, linkage="complete").fit(X)
    complete_labels = clustering_complete.labels_
    clustering_average = AgglomerativeClustering(n_clusters=i, linkage="average").fit(X)
    average_labels = clustering_average.labels_
    Z = sp.cluster.hierarchy.linkage(Y, method = "centroid")
    Z = sp.cluster.hierarchy.fcluster(Z, i, criterion = "maxclust")
    x_silhouette.append(i)
    silhouette_result_single = metrics.silhouette_score(X, single_labels, metric='euclidean')
    silhouette_result_complete = metrics.silhouette_score(X, complete_labels, metric='euclidean')
    silhouette_result_average = metrics.silhouette_score(X, average_labels, metric='euclidean')
    silhouette_result_centroid = metrics.silhouette_score(X, Z)
    y_single_silhouette.append(silhouette_result_single)
    y_complete_silhouette.append(silhouette_result_complete)
    y_average_silhouette.append(silhouette_result_average)
    y_centroid_silhouette.append(silhouette_result_centroid)

fig, (single_linkage,complete_linkage,average_linkage,centroids) = plt.subplots(1,4, figsize = (25,10))
fig.suptitle("single-linkage - complete-linkage - average-linkage - centroids")
single_linkage.plot(x_silhouette, y_single_silhouette, 'cyan')
complete_linkage.plot(x_silhouette, y_complete_silhouette, 'red')
average_linkage.plot(x_silhouette, y_average_silhouette, 'green')
centroids.plot(x_silhouette, y_centroid_silhouette, 'orange')




#Goodness of clustering using different metrics
#Silhouette Coefficient
silhouette_result = metrics.silhouette_score(X, single_labels, metric='euclidean')
print("Silhouette Result: ", silhouette_result)

print("-----------------------------------------------------------------------------------")

#EXERCISE 3
#Random shuffle
print("\nPERFORMANCE AFTER SHUFFLE")
np.random.shuffle(X);
clustering_single = AgglomerativeClustering(n_clusters=k, linkage="single").fit(X)
single_labels = clustering_single.labels_
silhouette_result = metrics.silhouette_score(X, single_labels, metric='euclidean')
print("Silhouette Result with random shuffle: ", silhouette_result)

print("-----------------------------------------------------------------------------------")
#EXERCISE 4
#################################################################################################
# 4A
# The data is loaded from the csv file
df_cows = pd.read_csv("cows.csv")

#Initially, the column with the names is removed
df_cows = df_cows.drop(columns = ['name'])
df_cows_num = df_cows.drop(columns = ['race', 'character', 'music'])

#Create dictionaries for replacing non-numerical features
race = {'holstein': 1, 'ayrshire': 2, 'finncattle': 3}
character = {'lively': 1, 'kind': 2, 'calm': 3}
music = {'rock': 1, 'country': 2, 'classical': 3}

df_cows.race = [race[item] for item in df_cows.race]
df_cows.character = [character[item] for item in df_cows.character]
df_cows.music = [music[item] for item in df_cows.music]

#print(df_cows_num)
df_cows_num_array = np.array(df_cows_num)

#Normalization of the values between 0 and 1
df_cows_num_array = (df_cows_num_array-df_cows_num_array.min(axis=0)) / (df_cows_num_array.max(axis=0)-df_cows_num_array.min(axis=0))
euclidean_matrix = np.zeros((len(df_cows_num_array), len(df_cows_num_array)))

for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
        euclidean_dist = math.sqrt((df_cows_num_array[j][0]-df_cows_num_array[i][0])**2 + (df_cows_num_array[j][1]-df_cows_num_array[i][1])**2)
        euclidean_matrix[i][j] = euclidean_dist

#Euclidean Matrix
print("\EUCLIDEAN MATRIX")
print(euclidean_matrix)


print("-----------------------------------------------------------------------------------")
#################################################################################################
# 4B
#Categorical data

df_cows = pd.read_csv("cows.csv")
df_cows_cat = df_cows.drop(columns = ['name', 'age', 'milk'])
df_cows_cat_array = np.array(df_cows_cat)


#Getting frequency of the features to compute the goodall mesaure
sum_race = 0
sum_character = 0
sum_music = 0

#Frequency_vector contains the repetitions for each feature and for each cow, so in the
#first case, for the first cow, there are 2 cows with the same race, 1 lively cow and 2 cows
#who prefer rock music
frequency_vector = np.zeros((len(df_cows_cat_array),3))

for i in range(len(df_cows_cat_array)):
    sum_race = 0
    sum_music = 0
    sum_character = 0
    for j in range(len(df_cows_cat_array)):
        if (df_cows_cat_array[i][0] == df_cows_cat_array[j][0]):
            sum_race = sum_race +1
        if (df_cows_cat_array[i][1] == df_cows_cat_array[j][1]):
            sum_character = sum_character +1
        if (df_cows_cat_array[i][2] == df_cows_cat_array[j][2]):
            sum_music = sum_music + 1
        
        frequency_vector[i] = [sum_race/6, sum_character/6, sum_music/6]

#print(frequency_vector)


#Each element in the overlap matrix represent p(xi), so the element (row 3, column 4), represents how many things 
#3 and 4 have in common, out of 3, meaning the overlap. In this example, they have in common the character (calm) and the music
#(classical), so 2/3
music = 0
character = 0
race = 0

overlap_matrix = np.zeros((len(df_cows_cat_array), len(df_cows_cat_array)))

#Shared values matrix is a matrix that has 5 matrices inside with 5 rows and 3 columns. Each of them is the individual
#table for each cow, so, for the first cow, the first row will be ones and then it will have another 1 in the first position of
#the third row (both have the same race) and another 1 in the third position of the second row (both like rock music)
shared_values_matrix = np.zeros((len(df_cows_cat_array), len(df_cows_cat_array), 3))
#Shared values vector summarize the information in het previous matrix, so for the first cow, the first feature is shared twice (2
#cows with the same race), the second feature is unique (it is the only lively cow) and two cows like rock music
shared_values_vector = np.zeros((len(df_cows_cat_array), 3))
race_sum = 0
character_sum = 0
music_sum = 0


for i in range(len(df_cows_cat_array)):
    race_sum = 0
    character_sum = 0
    music_sum = 0   
    for j in range(len(df_cows_cat_array)):
        goodall_num = 0
        if(df_cows_cat_array[i][2] == df_cows_cat_array[j][2]):
            music = 1
            music_sum = music_sum +1
        if(df_cows_cat_array[i][1] == df_cows_cat_array[j][1]):
            character = 1
            character_sum = character_sum +1
        if(df_cows_cat_array[i][0] == df_cows_cat_array[j][0]):
            race =  1 
            race_sum = race_sum+1
        overlap_matrix[i][j] = (music + character + race)/3
        shared_values_matrix[i][j] = [race, character, music]
        shared_values_vector[i] = [race_sum, character_sum, music_sum]
        
        music = 0
        character = 0
        race = 0

goodall_matrix = np.zeros((len(df_cows_cat_array), len(df_cows_cat_array)))
goodall_num = 0

for i in range(len(df_cows_cat_array)):
    for j in range(len(df_cows_cat_array)):
        goodall_num = 0
        for k in range(3):
            if i != j:
                if shared_values_matrix[i][j][k] == 1:
                    goodall_num = goodall_num + 1 - frequency_vector[i][k]**2
            elif i == j:
                if shared_values_matrix[i][j][k] == 1:
                    goodall_num = goodall_num + 1 - frequency_vector[i][k]**2
                    
            goodall_matrix[i][j] = goodall_num/3

print("\nGOODALL MATRIX")
print(goodall_matrix)

#################################################################################################
# 4C
print("-----------------------------------------------------------------------------------")
#Similarity in mixed data
df_cows = pd.read_csv("cows.csv")
#Initially, the column with the names is removed
df_cows = df_cows.drop(columns = ['name'])
df_cows_num = df_cows.drop(columns = ['race', 'character', 'music'])
df_cows_num_array = np.array(df_cows_num)
#Comupte lambda as the fraction of numerical features in data
df_cows_array = np.array(df_cows)

numerical = 0
non_numerical = 0
for i in range(len(df_cows_array)):
    for j in range(5): 
        if isinstance(df_cows_array[i][j], int):
            numerical = numerical + 1
        else:
            non_numerical = non_numerical + 1


lambda_var = numerical / (non_numerical + numerical)

#For the numerical values, it is necessary to compute a similarity. The euclidean distance
#was previously computed. Now, the cosine similarity is calculated using sklearn library
cosine_matrix = np.zeros((len(df_cows_num_array), len(df_cows_num_array)))
for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
            cosine_matrix[i][j] = metrics.pairwise.cosine_similarity([df_cows_num_array[i]], [df_cows_num_array[j]])

print("\nCOSINE SIMILARITY MATRIX")
print(cosine_matrix)

#Similarity in numerical data: cosine similarity
#Similarity in non-numerical data: overlap similarity

#Reshape matrix into 1D array
cosine_similarity_array = np.zeros(36)
overlap_similarity_array = np.zeros(36)
index = 0
for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
        cosine_similarity_array[index] = cosine_matrix[i][j]
        overlap_similarity_array[index] = overlap_matrix[i][j]        
        index = index + 1
        
stdev_num = statistics.stdev(cosine_similarity_array)
stdev_cat = statistics.stdev(overlap_similarity_array)


#Mixed data similarity measure
mixed_similarity_matrix = np.zeros((len(df_cows_num_array), len(df_cows_num_array)))
for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
        mixed_similarity_matrix[i][j] = (lambda_var)*cosine_matrix[i][j]/stdev_num + (1-lambda_var)*overlap_matrix[i][j]/stdev_cat
        
print("\nMIXED DATA SIMILARITY MATRIX")
print(mixed_similarity_matrix)


#Different approach: computing the similarity matrix linked to the euclidean distance
units_matrix = [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]]
euclidean_similarity_matrix = units_matrix - euclidean_matrix
print("\nEUCLIDEAN SIMILARITY MATRIX")
print(euclidean_similarity_matrix)

#Mixed data similarity measure
mixed_similarity_matrix_2 = np.zeros((len(df_cows_num_array), len(df_cows_num_array)))
for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
        mixed_similarity_matrix_2[i][j] = (lambda_var)*euclidean_similarity_matrix[i][j]/stdev_num + (1-lambda_var)*overlap_matrix[i][j]/stdev_cat
        
print("\nNEW MIXED DATA SIMILARITY MATRIX")
print(mixed_similarity_matrix_2)

#Expressed as a distance
mixed_distance_matrix = np.zeros((len(df_cows_num_array), len(df_cows_num_array)))
for i in range(len(df_cows_num_array)):
    for j in range(len(df_cows_num_array)):
        mixed_distance_matrix[i][j] = (1-mixed_similarity_matrix_2[i][j])/mixed_similarity_matrix_2[i][j]

print("\nMIXED DATA DISTANCE MATRIX")
print(mixed_distance_matrix)