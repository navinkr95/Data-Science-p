# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:10:17 2022

@author: Navin
"""

# K-Means Clustering

# Projects: wine Segmentation
# A Company wants to identify segments of Wine for targetted marketing.

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\SKILLEDGES PUNE\Datasets\wine.csv')
X = dataset[[" Alcohol","Color intensity"]]

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Here I got the optimal no of cluster is 3.Now I will randomly
# distribute the no of cluster in the dataset. By the help of this, It will
#  divide the wine into 3 non-overlapping homogeneous group based on alcohol
#  and color intensity.


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# With the help of fit_predict, we got to know that which
# data come under which cluster


kmeans = pd.DataFrame(y_kmeans)
dataset_1 = pd.concat([dataset,kmeans],axis=1)

# From dataset_1, we got to know that,
# All the data present in index number 0,comes under 3rd cluster.
# All the data present in index number 3, comes under 2nd cluster
# All the data present in index number 23, comes under 1st cluster and so on
# By this, We can easily observse that which data come under which cluster.

# But it is not posssible to observe all the data through this.
# So it always a better way to observe the data through visualisation.

# Visualising the clusters
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s = 100, c = 'red' , label = 'Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s = 100, c = 'blue' , label = 'Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s = 100, c = 'green' , label = 'Cluster 3')

plt.title('Clusters of wines')
plt.xlabel('Alcohol content')
plt.ylabel('color intensity')
plt.legend()
plt.show()

# By Analysing the data,
#Whose color intensity is greater than 1 and < 4,that data comes under 1st cluster
#Whose color intensity is greater than 4 and < 7,that data comes under 2nd cluster
#Whose color intensity is greater than 7 and <= 13,that data comes under 3nd cluster


# Hierarchical Clustering

# Method 1 "Ward"

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distances')


#cut the dendrogram tree with a horizontal line at a height where the line can
# traverse without intersecting the merging point. Hence, we can see the ideal 
# no. of clusters is 3.

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# With the help of fit_predict, we got to know that which
# data come under which cluster

# All the data present in index number 0,comes under 2nd cluster.
# All the data present in index number 3, comes under 1st cluster
# All the data present in index number 59, comes under 3rd cluster and so on
# By this, We can easily observse that which data comes under which cluster.

# But it is not posssible to observe all the data through this.
# So it always a better way to observe the data through visualisation.

# Visualising the clusters
plt.scatter(X.iloc[y_hc == 0, 0], X.iloc[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X.iloc[y_hc == 1, 0], X.iloc[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X.iloc[y_hc == 2, 0], X.iloc[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.title('Clusters of wines')
plt.xlabel('Alcohol content')
plt.ylabel('color intensity')
plt.legend()

# By Analysing the data,
#Whose color intensity is greater than 6 and < =13,that data come under 1st cluster
#Whose color intensity is greater than 4 and < 6,that data come under 2nd cluster
#Whose color intensity is greater than 1 and < 4,that data come under 3nd cluster


# Method 2 "Average"

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'average'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distances')

#cut the dendrogram tree with a horizontal line at a height where the line can
# traverse without intersecting the merging point. Hence, we can see the ideal 
# no. of clusters is 4.

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'average')
y_hc = hc.fit_predict(X)

# With the help of fit_predict, we got to know that which
# data come under which cluster

# All the data present in index number 0,comes under 1st cluster.
# All the data present in index number 18, comes under 2nd cluster
# All the data present in index number 59, comes under 4th cluster
# All the data present in index number 158,comes under 3rd cluster and so on.

# By this, We can easily observse that which data comes under which cluster.

# But it is not posssible to observe all the data through this.
# So it always a better way to observe the data through visualisation.


# Visualising the clusters

plt.scatter(X.iloc[y_hc == 0, 0], X.iloc[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X.iloc[y_hc == 1, 0], X.iloc[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X.iloc[y_hc == 2, 0], X.iloc[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X.iloc[y_hc == 3, 0], X.iloc[y_hc == 3, 1], s = 100, c = 'black', label = 'Cluster 4')

plt.title('Clusters of wines')
plt.xlabel('Alcohol content')
plt.ylabel('color intensity')
plt.legend()

# By Analysing the data,
#Whose color intensity is greater than 3.5 and < 8,that data come under 1st cluster
#Whose color intensity is greater than 8 and < 12,that data come under 2nd cluster
#Whose color intensity is 13,that data come under 3nd cluster
#Whose color intensity is greater than 1 and < 4,that data come under 3nd cluster


# Method 3 "complete"

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distances')

#cut the dendrogram tree with a horizontal line at a height where the line can
# traverse without intersecting the merging point. Hence, we can see the ideal 
# no. of clusters is 4.

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
y_hc = hc.fit_predict(X)

# With the help of fit_predict, we got to know that which
# data come under which cluster

# All the data present in index number 0,comes under 1st cluster.
# All the data present in index number 3, comes under 2nd cluster
# All the data present in index number 158, comes under 3rd cluster and so on.

# By this, We can easily observse that which data comes under which cluster.

# But it is not posssible to observe all the data through this.
# So it always a better way to observe the data through visualisation.


# Visualising the clusters

plt.scatter(X.iloc[y_hc == 0, 0], X.iloc[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X.iloc[y_hc == 1, 0], X.iloc[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X.iloc[y_hc == 2, 0], X.iloc[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')


plt.title('Clusters of wines')
plt.xlabel('Alcohol content')
plt.ylabel('color intensity')
plt.legend()

# By Analysing the data,
#Whose color intensity is greater than 1 and < 7,that data comes under 1st cluster
#Whose color intensity is greater than 7 and < 10.8,that data comes under 2nd cluster
#Whose color intensity is greater than 11.8 and <= 13, that data comes under 3rd cluster