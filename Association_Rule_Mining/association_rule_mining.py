import pandas as pd
import numpy as np
from itertools import combinations
import sys

# Randomly genrated data set for testing
#df = pd.DataFrame(np.random.randint(0,2,size=(100, 24)), columns=list(string.ascii_lowercase[:24]))

# Loading dataset
if int(sys.argv[1]) == 2:
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data", sep=",", header=None,
                     names=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health",
                            "acceptance"])
elif int(sys.argv[1]) == 3:
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", sep=",",
                     header=None, names=["cap-shape", "cap-surface", "cap-color", "is_bruises", "odor", "gill-attachment",
                            "gill-spacing", "gill_size","gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                            "stalk_surface_below_ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil_type",
                            "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat","mushroom_type"])
    df = df.replace('?', np.NaN).dropna()
else :
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", sep=",", header=None,
                     names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "condition"])

df = pd.get_dummies(df)
df = df.reindex(sorted(df.columns), axis=1)


# Declaring various dictionaries to store itemset
itemset = {}
itemset.update({1: {}, 2:{}, 3: {}, 4: {}})
freq_itemset = {}
freq_itemset.update({1: {}, 2:{}, 3: {}, 4: {}})
infreq_itemset = {}
infreq_itemset.update({1: {}, 2:{}, 3: {}, 4: {}})
min_s = int(0.01 * int(sys.argv[3]) * len(df))
global candidate_items
candidate_items = 0
rules = {}
lift = {}

# Counting Total frequent itemset
def freq_itemset_count(itemset_dict):
    count = 0
    for itemsets in itemset_dict:
        count += len(itemset_dict[itemsets])
    return count


# Printing top five rules:
def print_rules(rules):
    for key in  sorted(rules, key=lambda x:rules[x],reverse=True)[:5]:
        print "{0}     {1}".format(key,str(rules[key]))


# Brute force rule generation count
def brute(freq_itemset):
    count = 0
    for itemset in freq_itemset:
        count += len(freq_itemset[itemset])*((2**itemset)-2)
    return count


# Creating frequent 1-itemset
for i in df.columns:
    candidate_items+=1
    itemset[1][(i,)] = 0
for i in range(len(df)):
    for j in itemset[1]:
        if df.iloc[i].loc[j] == 1:
            itemset[1][j]+=1
for key in itemset[1]:
    if itemset[1][key] < min_s:
        infreq_itemset[1][key] = itemset[1][key]
    else:
        freq_itemset[1][key] = itemset[1][key]


# Itemset creation using Fk-1 and F1 method
def f_k_1(freq_itemset, n=1):
    global candidate_items
    for i in freq_itemset[n]:
        for j in freq_itemset[1]:
            if i[-1] < j[-1]:
                temp = list(i)
                temp.append(''.join(j))
                candidate_items +=1
                itemset[n+1][tuple(temp)] = 0
    del_item = []
    for j in itemset[n + 1]:
        for i in infreq_itemset[n]:
            if set(i).issubset(j):
                infreq_itemset[n+1][j] = 0
                del_item.append(j)
                break
    for item in del_item:
        del itemset[n+1][item]
    for i in range(len(df)):
        master_list = []
        for j in df.columns:
            if df.iloc[i].loc[j] == 1:
                master_list.append(j)
        for j in itemset[n + 1]:
            if set(j).issubset(master_list):
                itemset[n+1][j] += 1
    for key in itemset[n+1]:
        if itemset[n+1][key] < min_s:
            infreq_itemset[n+1][key] = itemset[n+1][key]
        else:
            freq_itemset[n+1][key] = itemset[n+1][key]
    if n+1 < 4 :
        f_k_1(freq_itemset, n+1)



# Itemset creation using Fk-1 and Fk-1 method
def f_k_k(freq_itemset, n=1):
    global candidate_items
    for i in freq_itemset[n]:
        for j in freq_itemset[n]:
            if i[:-1] == j[:-1]:
                if i[-1] < j[-1]:
                    temp = list(i)
                    temp.append(''.join(j[-1]))
                    candidate_items+=1
                    itemset[n+1][tuple(temp)] = 0
    del_item = []
    for j in itemset[n + 1]:
        for i in infreq_itemset[n]:
            if set(i).issubset(j):
                infreq_itemset[n+1][j] = 0
                del_item.append(j)
                break
    for item in del_item:
        del itemset[n+1][item]
    for i in range(len(df)):
        master_list = []
        for j in df.columns:
            if df.iloc[i].loc[j] == 1:
                master_list.append(j)
        for j in itemset[n + 1]:
            if set(j).issubset(master_list):
                itemset[n+1][j] += 1
    for key in itemset[n+1]:
        if itemset[n+1][key] < min_s:
            infreq_itemset[n+1][key] = itemset[n+1][key]
        else:
            freq_itemset[n+1][key] = itemset[n+1][key]
    if n+1 < 4 :
       f_k_k(freq_itemset, n+1)


# Maximal and closed frequent set
def max_and_closed_freq_set(freq_itemset):
    max_set = list(freq_itemset[len(freq_itemset)].keys())
    closed_set = list(freq_itemset[len(freq_itemset)].keys())
    for i in range(3,1,-1):
        for key in freq_itemset[i]:
            max_ind = True
            closed_ind = True
            for superset_key in freq_itemset[i+1]:
                if set(key).issubset(superset_key):
                    max_ind = False
                    if freq_itemset[i][key] == freq_itemset[i+1][superset_key]:
                        closed_ind = False
                        break
            if max_ind:
                max_set.append(key)
            if closed_ind:
                closed_set.append(key)
    return max_set, closed_set


# Generating Rules using confidence
def rule_generation():
    for m in range(4,1,-1):
        for item in freq_itemset[m]:
            reduce = []
            for i in range(len(item)-1):
                comb = list(combinations(item, len(item)-i-1))
                for j in reduce:
                    for k in comb:
                        if set(j).issubset(tuple(set(item)-set(k))):
                            comb.remove(k)
                for j in comb:
                    conf = freq_itemset[len(item)][item]/float(freq_itemset[len(j)][j])
                    if conf < min_c:
                        reduce.append(tuple(set(item)-set(j)))
                        break
                    rules[' ^ '.join(j)+' => '+' ^ '.join(set(item)-set(j))] = conf


# Generating rules based on lift
def rule_generation_lift():
    for m in range(4,1,-1):
        for item in freq_itemset[m]:
            reduce = []
            for i in range(len(item)-1):
                comb = list(combinations(item, len(item)-i-1))
                for j in reduce:
                    for k in comb:
                        if set(j).issubset(tuple(set(item)-set(k))):
                            comb.remove(k)
                for j in comb:
                    conf = freq_itemset[len(item)][item]/float(freq_itemset[len(j)][j])
                    l = (len(df) *conf)/freq_itemset[len(tuple(set(item)-set(j)))][tuple(sorted(tuple(set(item)-set(j))))]
                    if l < 1.4:
                        reduce.append(tuple(set(item)-set(j)))
                        break
                    lift[' ^ '.join(j)+' => '+' ^ '.join(set(item)-set(j))] = round(l,2)



if int(sys.argv[2]) == 2:
    f_k_k(freq_itemset)
    print "For Fk-1 * Fk-1:"
    print "No. of candidate itemset: " + str(candidate_items)
    print "No. of  of frequent itemse: " + str(freq_itemset_count(freq_itemset))
else:
    f_k_1(freq_itemset)
    print "For Fk-1 * F1:"
    print "No. of candidate itemset: " +str(candidate_items)
    print "No. of  of frequent itemse: " +str(freq_itemset_count(freq_itemset))


maximum_set, closed_set = max_and_closed_freq_set(freq_itemset)
print "Number of Maximal Frequent Itemset: " +str(len(maximum_set))
print "Number of Frequent Closed Itemset: " +str(len(closed_set))


min_c = 0.01 * int(sys.argv[4])

rule_generation()
rule_generation_lift()
print "No of rules generated by brute-force: "+str(brute(freq_itemset))
print "No of rules generated by Confidence Pruning: "+str(len(rules))
print
print "Top Five rules with confidence:"
print_rules(rules)
print
print "Top Five rules with Lift"
print_rules(lift)

