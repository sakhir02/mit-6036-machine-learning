#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Implement perceptron, average perceptron, and pegasos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb
import itertools
import operator
import functools


# In[53]:


A=np.array([[1,2],[1,3],[1,4],[1,6]])
print(A)
print(np.shape(A[:,1:2])[0])
print((A[:,0]))
l1=[1,2,4,6]
l2=[[]]
l2[0]=l1
#print(listl)
print(l2)
import numpy as np
def rv(value_list):
    return np.array(value_list).reshape(1,len(value_list))
    

print(rv(l1))


# In[4]:


import numpy as np
def index_final_col(A):
     return A[:, -1].reshape(-1, 1)
A=np.array([[2,4,5],[2,5,1],[4,6,1]])
print(A)
print(A[:, -1])#bree
print(A[:, np.shape(A)[1]-1])
print(index_final_col(A))
np.shape(index_final_col(A))


# In[50]:


import numpy as np
def positive(x, th, th0):
    if (np.dot(np.transpose(th),x)+th0)[-1]>0:
        return np.array([[1]])
    elif (np.dot(np.transpose(th),x)+th0)[-1]==0:
        return np.array([[0]])
    else:
        return np.array([[-1]])

            
data=np.array([[2,4,5],[2,5,1],[4,6,1]])
data[:,:]


# In[173]:


# Test data for problem 2.1
data1, labels1, data2, labels2 = (np.array([[-2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524,
         3.10363716,  2.13434789,  1.61328413,  2.10491257, -3.87099125,
         3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746,
        -4.93242615,  2.16880073, -4.34923279, -0.76154262,  3.04879591,
        -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576,
        -1.00326657,  3.62331703, -4.97864369, -3.31037331, -1.16371314],
       [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211,
         2.4843758 , -3.47935105, -4.31857472, -0.11863976,  0.34441625,
         0.77851176,  1.6403079 , -0.57558913, -3.62293005, -2.9638734 ,
        -2.80071438,  2.82523704,  2.07860509,  0.23992709,  4.790368  ,
        -2.33037832,  2.28365246, -1.27955206, -0.16325247,  2.75740801,
         4.48727808,  1.6663558 ,  2.34395397,  1.45874837, -4.80999977]]), 
 np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.,
        -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.,
        -1., -1., -1., -1.]]), np.array([[ 0.6894022 , -4.34035772,  3.8811067 ,  4.29658177,  1.79692041,
         0.44275816, -3.12150658,  1.18263462, -1.25872232,  4.33582168,
         1.48141202,  1.71791177,  4.31573568,  1.69988085, -2.67875489,
        -2.44165649, -2.75008176, -4.19299345, -3.15999758,  2.24949368,
         4.98930636, -3.56829885, -2.79278501, -2.21547048,  2.4705776 ,
         4.80481986,  2.77995092,  1.95142828,  4.48454942, -4.22151738],
       [-2.89934727,  1.65478851,  2.99375325,  1.38341854, -4.66701003,
        -2.14807131, -4.14811829,  3.75270334,  4.54721208,  2.28412663,
        -4.74733482,  2.55610647,  3.91806508, -2.3478982 ,  4.31366925,
        -0.92428271, -0.84831235, -3.02079092,  4.85660032, -1.86705397,
        -3.20974025, -4.88505017,  3.01645974,  0.03879148, -0.31871427,
         2.79448951, -2.16504256, -3.91635569,  3.81750006,  4.40719702]]),
 np.array([[-1., -1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1.,  1.]]))


# In[176]:


import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    n_features= np.shape(data)[0]
    n_points=np.shape(data)[1]
    theta=np.zeros((n_features,1))
    theta0=np.zeros((1,1))

    for j in range(T):
        for i in range(n_points):
            xi=data[:,i:i+1]
            yi=labels[:,i]
            if (yi*(np.dot(theta.T,xi)+theta0)<=0):
                theta= theta+ xi*yi
                theta0=theta0 + yi
            #if hook: hook((theta, theta0))
            
    return theta, theta0
th,th0 = perceptron(data1,labels1)

def score_mat(data, labels, ths, th0s):
   pos = np.sign(np.dot(ths.T, data) + np.transpose(th0s))
   return np.sum(pos == labels, axis = 1, keepdims = True)

score_mat(data1,labels1,th,th0)

np.split(data1,5,axis=1)[-1]

import numpy as np

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    n_features= np.shape(data)[0]
    n_points=np.shape(data)[1]
    theta=np.zeros((n_features,1))
    thetas=np.zeros((n_features,1))
    thetas0=np.zeros((1,1))
    theta0=np.zeros((1,1))
    count=0
    for j in range(T):
        for i in range(n_points):
            xi=data[:,i:i+1]
            yi=labels[:,i]
            if (yi*(np.dot(theta.T,xi)+theta0)<=0):
                theta= theta+ xi*yi
                theta0=theta0 + yi
            thetas=thetas +theta
            thetas0=thetas0 + theta0
            count=count + 1
            #if hook: hook((thetas, thetas0))
    return thetas/(n_points*T), thetas0/(n_points*T), count
import numpy as np

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0=learner(data_train, labels_train, params={}, hook=None)
    return score_mat(data_test,labels_test,th,th0)/data_test.shape[1]
    

    import numpy as np
def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    total = 0
    
    for _ in range(it):
        datatrain = data_gen(n_train)
        datatest  = data_gen(n_test)
        total= total + eval_classifier(learner, datatrain[0], datatrain[1], datatest[0], datatest[1])
    
    return total / it

import numpy as np

    
def xval_learning_alg(learner, data, labels, k):
    data_splits   = np.array_split(data,   k, axis=1)
    labels_splits = np.array_split(labels, k, axis=1)
    scores = []

    for i in range(k):
        # fold i = test
        data_test   = data_splits[i]
        labels_test = labels_splits[i]

        # tout le reste = train
        data_train   = np.concatenate([data_splits[j]   for j in range(k) if j != i], axis=1)
        labels_train = np.concatenate([labels_splits[j] for j in range(k) if j != i], axis=1)

        # évaluer
        score = eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        scores.append(score)

    return np.mean(scores)


# In[100]:


import numpy as np
# data is dimension d by n
# labels is dimension 1 by n
# ths is dimension d by m
# th0s is dimension 1 by m
# return matrix of integers indicating number of data points correct for
# each separator:  dimension m x 1
def score_mat(data, labels, ths, th0s):
   pos = np.sign(np.dot(ths.T, data) + np.transpose(th0s))
   return np.sum(pos == labels, axis = 1, keepdims = True)

def best_separator(data, labels, ths, th0s):
   best_index = np.argmax(score_mat(data, labels, ths, th0s))
   return cv(ths[:,best_index]), th0s[:,best_index:best_index+1]


# In[108]:


def score_mat(data, labels, ths, th0s):
   pos = np.sign(np.dot(ths.T, data) + np.transpose(th0s))
   return np.sum(pos == labels, axis = 1, keepdims = True)

def gen_lin_separable(num_points=20, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    ''' 
    Generate linearly separable dataset X, y given theta and theta0
    Return X, y where
    X is a numpy array where each column represents a dim-dimensional data point
    y is a column vector of 1s and -1s
    '''
    X = np.random.uniform(low=-5, high=5, size=(dim, num_points))
    y = np.sign(np.dot(np.transpose(th), X) + th_0)
    return X, y

def gen_flipped_lin_separable(num_points=20, pflip=0.25, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    '''
    Generate difficult (usually not linearly separable) data sets by
    "flipping" labels with some probability.
    Returns a method which takes num_points and flips labels with pflip
    '''
    def flip_generator(num_points=20):
        X, y = gen_lin_separable(num_points, th, th_0, dim)
        flip = np.random.uniform(low=0, high=1, size=(num_points,))
        for i in range(num_points):
            if flip[i] < pflip: y[0,i] = -y[0,i]
        return X, y
    return flip_generator

data_gen = gen_flipped_lin_separable(pflip=0.25) 
print(eval_learning_alg(perceptron,data_gen,20,20,50))
print(eval_learning_alg(averaged_perceptron,data_gen,20,20,50))


# In[121]:


def new_eval_learning_alg(learner, data, n_traintest, it):
    total = 0
    
    for i in range(it):
        datatrain = data(n_traintest)
        total= total + eval_classifier(learner, datatrain[0], datatrain[1], datatrain[0], datatrain[1])
    
    return total / it
data = gen_flipped_lin_separable(pflip=0.25) 
print(new_eval_learning_alg(perceptron,data,20,50))
print(new_eval_learning_alg(averaged_perceptron,data,20,50))


# In[333]:


import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    n_features= np.shape(data)[0]
    n_points=np.shape(data)[1]
    theta=np.zeros((n_features,1))
    #count=0
    theta0=np.zeros((1,1))

    for j in range(T):
        for i in range(n_points):
            xi=data[:,i:i+1]
            yi=labels[:,i]
            if (yi*(np.dot(theta.T,xi)+theta0)<=0):
                theta= theta+ xi*yi
                theta0=theta0 + yi
                #count=count +1 
               
            #if hook: hook((theta, theta0))
            
    return theta,theta0, #count

#data =   np.array([[2, 3,  4,  5]])
#labels = np.array([[1, 1, -1, -1]])
#data=np.array([[0.2, 0.8, 0.2, 0.8],[0.2,  0.2,  0.8,  0.8],[1.,1.,1.,1.]])
#labels =np.array([[-1, -1, 1, 1]])
#averaged_perceptron(data,labels,params={'T':800000})
#perceptron(data,labels,params={'T': 200000})
#len(data)


# In[283]:


import numpy as np

def one_hot(x, k):
    assert 1 <= x <= k, "x={x} doit être entre 1 et {k}"
    p=np.zeros((k,1))
    p[x - 1] = 1
    return p

def one_hotList(features):
    features=features.flatten()
    y=np.max(features)
    encode =[]
    for i in features:
        encode.append(one_hot(int(i),y))
    return np.hstack(encode)

data =   np.array([[2, 3,  4,  5]])
labels = np.array([[1, 1, -1, -1]])
one_hotList(data) 
datax=one_hotList(data) 
perceptron(datax,labels,params={'T': 800000})
        
        


# In[308]:


data =   np.array([[1, 2, 3,  4,  5, 6]])
labels=np.array([[1, 1, -1, -1, 1, 1]])
def one_hotList(features, k):        # k explicite ✅
    features = features.flatten()
    encode = []
    for i in features:
        encode.append(one_hot(int(i), k))
    return np.hstack(encode)

datax = one_hotList(data, k=6)

theta, theta0  = perceptron(datax,labels,params={'T': 100})
print(theta.flatten())
print(theta0)


# In[ ]:





# In[ ]:




