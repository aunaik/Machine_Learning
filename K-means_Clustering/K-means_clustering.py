#!/usr/bin/env python
#Author: Akshay Naik

#Description: The basic K-means clustering that enables the use of various other distance functions. The algorithm
selects among: (i) Euclidean distance (default), (ii) cosine distance, (iii)Cityblock distance. In all cases the 
algorithm computes the centroid as the mean of data points that belongs to the cluster.

import numpy as np
import pandas as pd
import sys
from copy import deepcopy


# Objjective Function
def sum_of_square_error(data, clusters, c):
    SSE = 0
    for i in range(len(c)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i+1]
        if len(points) != 0:
            SSE += np.sum(np.sum(np.subtract(c[i],points)**2, axis = 1))
    return SSE


# Citiblock Distance
def citi_dist(x, c):
    return np.sum(abs(x - c), axis=1)


# Eucledian Distance
def e_dist(x, c, ax = None):
    return np.linalg.norm(x-c, axis= ax)


# Cosine Distance
def c_dist(x,c):
    norm = np.multiply(np.linalg.norm(x), np.linalg.norm(c, axis= 1, keepdims= True))
    return 1 - np.round((np.divide(np.dot(x, c.T).reshape(len(c),1),norm)).flatten(),7)


# Function 1
def fn_1_dist(x, c):
    temp_data = x - c
    temp_data_x = np.clip(temp_data, 0, None)
    temp_data_y = np.clip(-temp_data, 0, None)
    return np.sqrt(np.sum(temp_data_x, axis=1) ** 2 + np.sum(temp_data_y, axis=1) ** 2)


# Function 2
def fn_2_dist(x, c):
    temp_data_i = np.repeat(x.reshape(1, c.shape[1]), len(c), axis=0)
    temp_data = x - c
    temp_data_x = np.clip(temp_data, 0, None)
    temp_data_y = np.clip(-temp_data, 0, None)
    return np.sqrt(np.sum(temp_data_x, axis=1) ** 2 + np.sum(temp_data_y, axis=1) ** 2) / \
                np.sum(np.maximum.reduce([abs(temp_data_i), abs(c), abs(temp_data)]), axis=1)


# class name assignment:
def class_name_assignment(clusters):
    temp_dict = {}
    for i in range(k):
        points = [class1[j] for j in range(len(data)) if clusters[j] == i + 1]
        if len(points) != 0:
            temp_dict[i+1] = max(set(points), key=points.count)
    return map(lambda x : temp_dict[x], clusters)


# Calculating Accuracy
def accuracy(final_class):
    count = 0
    l = len(final_class)
    for i in range(l):
        if final_class[i] == class1[i]:
            count += 1
    return count*100.0/l


# Generating random points for centroid
def generate_centroids(k):
    c = []
    for i in range(k):
        #np.random.seed(20 * (i + 1))
        temp = data[np.random.choice(data.shape[0], 20, replace=False)]
        c.append(np.mean(temp, axis = 0))
    return np.array(c)


# main function
def main(data):
    SSE = 0
    c = generate_centroids(k)
    c_old = np.zeros(c.shape)
    clusters = np.zeros(len(data))
    diff_in_clust = e_dist(c,c_old)
    count = 0
    while diff_in_clust !=0:
        count+=1
        for i in range(len(data)):
            if distance == 'cityblock':
                dist = citi_dist(data[i], c)
            elif distance == 'cosine':
                dist = c_dist(data[i], c)
            elif distance == 'fn_1':
                dist = fn_1_dist(data[i], c)
            elif distance == 'fn_2':
                dist = fn_2_dist(data[i], c)
            else:
                dist = e_dist(data[i], c, ax=1)
            clusters[i] = 1 + np.argmin(dist)
        c_old = deepcopy(c)
        SSE = sum_of_square_error(data, clusters, c)
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i+1]
            if len(points) != 0:
                c[i] = np.mean(points, axis=0)
        diff_in_clust = e_dist(c, c_old)

    final_class = class_name_assignment(clusters)
    print "Accuracy is: " + str(round(accuracy(final_class),2))+"%"
    print "Number of iterations: " + str(count)
    print "Number of distance calculations: " + str(count * len(data) * k)
    print "SSE: " + str(SSE)


which_dataset = sys.argv[1]

# Iris database
if which_dataset == 'iris':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    dataset = pd.read_csv(url, header=None)
    data = dataset.loc[:, dataset.columns != dataset.columns[-1]]
    class1 = dataset.loc[:, dataset.columns == dataset.columns[-1]]
    # No of clusters
    k = 3

#  Balance Scale Database
elif which_dataset == 'balance_scale':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    dataset = pd.read_csv(url, header=None)
    data = dataset.loc[:, dataset.columns != dataset.columns[0]]
    class1 = dataset.loc[:, dataset.columns == dataset.columns[0]]
    # No of clusters
    k = 3

# Glass database
elif which_dataset == 'glass':
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    dataset = pd.read_csv(url, header=None)
    dataset = dataset.drop(dataset.columns[0], axis=1)
    data = dataset.loc[:, dataset.columns != dataset.columns[-1]]
    class1 = dataset.loc[:, dataset.columns == dataset.columns[-1]]
    # No of clusters
    k = 7


# Removing the class vairable and converting Dataframe to matrix
data = data.apply(pd.to_numeric)
data = data.as_matrix()
class1 = class1.as_matrix().flatten()

distance = sys.argv[2]

main(data)
