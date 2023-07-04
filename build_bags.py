import pandas as pd
import numpy as np
import math
from collections import Counter

def create_bags(X, y, g = 10, bagprob = 0.3, seed = 331):
    
    np.random.seed(seed)
    anom_idx = np.where(y==1)[0]
    norm_idx = np.where(y==0)[0]
    n_anom = len(anom_idx)
    n_norm = len(norm_idx)
    bags = []
    instance_labels = []
    m = np.shape(X)[1]
    while n_anom>0 or n_norm>0:
        p = np.random.uniform(0,1,1)[0]

        if p <= bagprob and n_anom>0:
            n_anom_instances = np.random.randint(1,g//2,1)[0]
            if n_anom_instances > n_anom:
                n_anom_instances = n_anom
            if g-n_anom_instances > n_norm:
                break;
            anom_instances = np.random.choice(anom_idx, n_anom_instances, replace = False)
            norm_instances = np.random.choice(norm_idx, g-n_anom_instances, replace = False)
            bags.append(np.concatenate((X[norm_instances,:], X[anom_instances,:])))
            instance_labels.append(np.concatenate((y[norm_instances], y[anom_instances])))

            n_anom -= n_anom_instances
            n_norm -= g-n_anom_instances
            anom_idx = anom_idx[~np.isin(anom_idx, anom_instances)]
            norm_idx = norm_idx[~np.isin(norm_idx, norm_instances)]
        else:
            if g > n_norm:
                break;
            norm_instances = np.random.choice(norm_idx, g, replace = False)

            bags.append(X[norm_instances,:])
            instance_labels.append(y[norm_instances])

            n_norm -= g
            norm_idx = norm_idx[~np.isin(norm_idx, norm_instances)]
        
    bags = np.array(bags).reshape(-1,g,m)
    instance_labels = np.array(instance_labels).reshape(-1,g)
    bags_labels = np.sum(instance_labels, axis = 1)
    bags_labels[bags_labels>0] = 1

    y_inst = instance_labels.reshape(-1)

    print("Bag summary:", Counter(bags_labels))

    return bags,bags_labels,y_inst

def get_yinst(y_inst, g, test_idx):
    y_inst_test = [] #np.zeros(g*len(test_idx), np.int)
    
    for idx in test_idx:
        y_inst_test.append(y_inst[idx*g:(idx+1)*g]) 
    
    return np.array(y_inst_test).reshape(-1)
