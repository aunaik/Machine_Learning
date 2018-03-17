#!/usr/bin/env python
#B565: Data Mining
#Author: Akshay Naik

#Description: Implemented the greedy algorithm that learns a classification tree given a data set. The code 
#assumes that all features are numerical and properly finds the best threshold for each split. Uses Gini and
#information gain, as specified by the user, it decides on the best attribute to split in every step. Stops
#growing the tree when all examples in a node belong to the same class or the remaining examples
#contain identical features.

import pandas as pd
import numpy as np
import copy
import sys


# Data Structure of each node
class createNode():
    def __init__(self):
        self.label = None
        self.condition = None
        self.child = {}


# Splitting data into test and train
def split_data(data):
    data = data.sample(frac=1).reset_index(drop=True)
    n= int(len(data)//2.0)
    train_data=data[:n]
    test_data=data[n:]
    return train_data, test_data


# Condition to stop the tree growth
def stopping_cond(data):
    # print data[data.columns[-1]].drop_duplicates()
    if len(data[data.columns[-1]].drop_duplicates()) == 1:
        return True
    elif len(data[data.columns.difference([data.columns[-1]])].drop_duplicates()) == 1:
        return True
    elif len(data) < 20:
        return True
    return False


# Getting indices where values of Y(target variable) changes
def change_index(data):
    index = []
    for i in range(len(data)-1):
        #print data[data.columns[-1]]
        #print data.iloc[i]
        #print len(data)
        #exit()
        if data[data.columns[-1]].iloc[i] != data[data.columns[-1]].iloc[i+1]:
            index.append(i)
    return index


# Calculating Entropy
def entropy_cal(data):
    entropy = 0
    counts = data[data.columns[-1]].value_count()
    values, counts = np.unique(data[data.columns[-1]], return_counts = True)
    freqs = counts.astype('float')/len(data)
    for p in freqs:
        if p != 0.0:
            entropy -= p * np.log2(p)
    return entropy


# Calculating Gini
def gini_cal(data):
    gini = 1
    values, counts = np.unique(data[data.columns[-1]], return_counts=True)
    freqs = counts.astype('float') / len(data)
    for p in freqs:
        if p != 0.0:
            gini -= p*p
    return gini


# Finding the best split of data
def find_best_split(data, col):
    eval = {}
    for i in col:
        temp_dict = {}
        if i != data.columns[-1]:
            data = data.sort_values(dataset.columns[i])
            indices = change_index(data)
            for j in indices:
                s = (float(data[i].iloc[j]) + float(data[i].iloc[j+1]))/2.0
                left_split = data[[i,data.columns[-1]]][data[i]<=s]
                right_split = data[[i, data.columns[-1]]][data[i] > s]
                if measure == 'entropy':
                    l_val = entropy_cal(left_split)
                    r_val = entropy_cal(right_split)
                else:
                    l_val = gini_cal(left_split)
                    r_val = gini_cal(right_split)
                temp_dict[j] = (len(left_split)/float(len(data)))*l_val + (len(right_split)/float(len(data)))*r_val
            index = min(temp_dict, key=lambda x: temp_dict[x])
            eval[i] = (index,temp_dict[index])
    best_split = min(eval, key=lambda x: eval[x][1])
    return best_split, eval[best_split][0], eval[best_split][1]


# Classifying the record/records when one of the stopping condition is satisfied
def classify(data):
    if len(data)!=0:
        count = data[data.columns[-1]].value_counts().to_dict()
        return max(count, key=lambda x:count[x])
    else:
        count = dataset[dataset.columns[-1]].value_counts().to_dict()
        return max(count, key=lambda x: count[x])


# Building the decision tree
def generate_tree(data, col, count):
    if stopping_cond(data) or count > 9:
        leaf = createNode()
        leaf.label = classify(data)
        return leaf
    else:
        attr, val, e_val = find_best_split(data, col)
        if measure == 'entropy':
            if e_val >= entropy_cal(data):
                leaf = createNode()
                leaf.label = classify(data)
                return leaf
        else:
            if e_val >= gini_cal(data):
                leaf = createNode()
                leaf.label = classify(data)
                return leaf
        data = data.sort_values(dataset.columns[attr])
        s = (data[attr].iloc[val] + data[attr].iloc[val + 1]) / 2.0
        node = createNode()
        node.label = attr
        node.condition = s
        left_split = data[data[attr] <= s]
        node.child['less'] = generate_tree(left_split, col, count+1)
        right_split = data[data[attr] > s]
        node.child['greater'] = generate_tree(right_split, col, count+1)
        return node

# This function is not been use in the final implementation of the decision tree
# pruning tree - It pruns the tree after it is grown completely
def prun_tree(node):
    if node.label is None:
        return node
    curr_child = []
    for key in node.child.keys():
        c = prun_tree(node.child[key])
        if c is not None:
            curr_child.append(c.label)
    if len(curr_child) == 2:
        if curr_child[0] == curr_child[1]:
            node.label = curr_child[0]
            node.child = {}
            return node
    return node

# This function is not been use in the final implementation of the decision tree
# pre-order printing of tree
def print_tree(node):
    if node is not None:
        print "(",
        print node.label,
        for key in node.child.keys():
            print_tree(node.child[key])
        print ")",
    return

# Predicting class of unknown object
def predict_class(node, data):
    if len(node.child)==0:
        return node.label
    if data[node.label] <= node.condition:
        return predict_class(node.child['less'], data)
    else:
        return predict_class(node.child['greater'], data)


# Classifying all the data that is given to this function
def classification(node, data):
    ndata = copy.deepcopy(data)
    temp = []
    for i in range(len(data)):
        temp.append(predict_class(node, data.iloc[i]))
    ndata['predicted'] = temp
    return ndata


# Evaluating the model using precision and recall
def eval1(data):
    y_actual = pd.Series(data[data.columns[-2]], name='Actual')
    y_pred = pd.Series(data[data.columns[-1]], name='Predicted')
    conf_mat = pd.crosstab(y_actual, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
    p = {}; r = {}
    for i in range(len(conf_mat)):
        p[conf_mat.columns[i]] = conf_mat.iloc[i,i].astype(float)/np.sum(conf_mat.iloc[:,i])
        r[conf_mat.columns[i]] = conf_mat.iloc[i,i].astype(float)/np.sum(conf_mat.iloc[i,:])
    return np.mean(p.values())*100, np.mean(r.values())*100

# Evaluating the model using Accuracy
def eval2(data):
    return sum(data[data.columns[-2]] == data[data.columns[-1]])*100/len(data)



def main():
    train_accuracy = [];
    test_accuracy = [];
    train_recall = [];
    test_recall = []
    train_precision = []
    test_precision = []
    for i in range(5):
        # Splitting test and train data
        train_data, test_data = split_data(dataset)

        # Function call to generate classification tree
        count = 0
        node = generate_tree(train_data, col, count)

        # Function call to prune tree
        # prun_tree(node)

        # Function call to print tree
        # print_tree(node)
        # print

        # Classifying the train data
        labelled_train_data = classification(node, train_data)
        train_accuracy.append(eval2(labelled_train_data))
        precision, recall = eval1(labelled_train_data)
        train_precision.append(precision)
        train_recall.append(recall)

        # Classifying the train data
        labelled_test_data = classification(node, test_data)
        test_accuracy.append(eval2(labelled_test_data))
        precision1, recall1 = eval1(labelled_test_data)
        test_precision.append(precision1)
        test_recall.append(recall1)

    # Evaluate the train data
    print "The Percentage Accuracy with Train Data is: " + str(np.mean(train_accuracy)) + "%"
    print "The Percentage Precision with Train Data is: " + str(np.mean(train_precision)) + "%"
    print "The Percentage Recall with Train Data is: " + str(np.mean(train_recall)) + "%"
    print

    # Evaluate the train data
    print "The Percentage Accuracy with Test Data is: " + str(np.mean(test_accuracy)) + "%"
    print "The Percentage Precision with Test Data is: " + str(np.mean(test_precision)) + "%"
    print "The Percentage Recall with Test Data is: " + str(np.mean(test_recall)) + "%"


which_dataset = sys.argv[1]
# which measure to use
if sys.argv[2] == 'information_gain':
    measure = 'entropy'
else:
    measure = 'gini'

# Iris database
if which_dataset == 'iris':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    dataset = pd.read_csv(url, header=None)

#  Haberman's Survival Database
elif which_dataset == 'haberman':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    dataset = pd.read_csv(url, header=None)

# Breast cancer wisconsin database
elif which_dataset == 'cancer':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    cancer = pd.read_csv(url, header=None)
    cancer = cancer.drop(cancer.columns[0], axis=1)
    cancer = cancer.dropna(axis=0, how='any')
    dataset = cancer.loc[cancer.ne('?').all(1),]
else:
    url = sys.argv[3]
    dataset = pd.read_csv(url, header=None)

# getting list of column header from pandas data frame
col = dataset.columns.values.tolist()


main()



