#!/usr/bin/env python
#Author: Akshay Naik

#Description:The K-means clustering Algorithm with Elkan’s acceleration based on triangle inequality 
#that is intended to minimize the sum of squared errors objective function. The algorithm takes the data
#set D and the number of clusters K as inputs and returns cluster assignments for each data point; i.e., 
#an integer between 1 and K. The algorithm also returns the number of iterations and distance calculations
#as well as the total sum of squared errors


import numpy as np
import pandas as pd
import sys
from copy import deepcopy


# Final dataset
dataset = pd.read_fwf('birch.txt')
data = dataset.as_matrix()


# Objjective Function
def sum_of_square_error(data, clusters, c):
    SSE = 0
    for i in range(len(c)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i +1]
        if len(points) != 0:
            SSE += np.sum(np.sum(np.subtract(c[i],points)**2, axis = 1))
    return SSE


# Declaring global count variable
global steps
steps = 0
# Eucledian Distance
def e_dist(x, c, ax = None):
    global steps
    steps+=1
    return np.linalg.norm(x-c, axis= ax)


# Generating random points for centroid
def generate_centroids(k):
    c = []
    for i in range(k):
        np.random.seed(20*(i+1))
        temp = data[np.random.choice(data.shape[0], 20, replace=False)]
        c.append(np.mean(temp, axis = 0))
    return np.array(c)
step = 0.67
# Calculating minimum distance between centroids
def cent_dist(c):
    s = []
    for i in range(len(c)):
        s.append(e_dist(c[i],c, ax=1))
    return np.array(s)


# Elkan's Algorithm
def elkan(data):
    global steps
    SSE = 0
    c = generate_centroids(k)
    c_old = np.zeros(c.shape)
    a = np.ones(len(data), dtype=np.int)
    u = np.full(len(data),np.inf)
    l = np.zeros([len(data),k])
    diff_in_clust = e_dist(c, c_old)
    count = 1

    # First iteration to initialize upper-bound and lower-bound
    for i in range(len(data)):
        dist = e_dist(data[i], c, ax=1)
        l[i,:] = dist
        a[i] = 1 + np.argmin(dist)
    u = np.min(l,axis=1)
    c_old = deepcopy(c)
    for i in range(k):
        points = [data[j] for j in range(len(data)) if a[j] == i + 1]
        if len(points) != 0:
            c[i] = np.mean(points, axis=0)
    delta = e_dist(c, c_old, ax=1)
    for i in range(len(data)):
        u[i] += delta[a[i] - 1]
        for j in range(len(c)):
            l[i, j] = max(0, l[i, j] - delta[j])

    # Actual conditions of elkan acceleration
    while diff_in_clust != 0:
        count +=1
        steps*= step
        s = cent_dist(c)
        for i in range(len(data)):
            if u[i] <= (sorted(s[a[i]-1])[1]/2):
                continue
            r = True
            for j in range(len(c)):
                z = max(l[i,j], s[a[i]-1,j]/2)
                if j+1 == a[i] or u[i] <= z:
                    continue
                if r:
                    d1 = e_dist(data[i],c[a[i]-1])
                    r = False
                if d1 <= z:
                    continue
                d2 = e_dist(data[i], c[j])
                if d2 < d1:
                    a[i] = j + 1
        c_old = deepcopy(c)
        SSE = sum_of_square_error(data, a, c)

        # Updating the centers
        for i in range(k):
            points = [data[j] for j in range(len(data)) if a[j] == i+1]
            if len(points) != 0:
                c[i] = np.mean(points, axis=0)
        delta = e_dist(c, c_old, ax = 1)

        # Updating the upper bound and lower bound
        for i in range(len(data)):
            u[i] += delta[a[i]-1]
            for j in range(len(c)):
                l[i,j] = max(0, l[i,j]-delta[j])
        diff_in_clust = e_dist(c, c_old)

    print a
    print "Number of iterations: " + str(count)
    print "Number of distance calculations: " + str(int(steps))
    print "SSE: " + str(SSE)

k = int(sys.argv[1])

elkan(data)