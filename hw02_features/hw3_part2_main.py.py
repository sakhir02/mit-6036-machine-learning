#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pdb
import numpy as np
from functools import partial
import sys
import os
# Remplace par le vrai chemin vers tes fichiers
sys.path.append(r'C:\Users\HP\Desktop\DriveIA')  # Windows
import code_for_hw3_part2 as hw3 


#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
#we starte by loading our data set in python, by using load_auto_data function in hw3 part2, and then, make a choice about how our
#features gonna be represented, by using features function in hw3 too, 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# we start by importing our data set 
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
os.chdir(r'C:\Users\HP\Desktop\DriveIA')

mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    raise Exception("implement me!")

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_samples = x.shape[0]
    return x.reshape(n_samples, -1).T 


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    raise Exception("modify me!")


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    raise Exception("modify me!")

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data


# In[73]:





# In[19]:


auto_labels


# In[80]:


learner1=partial(hw3.averaged_perceptron, params={'T': 10})


# In[68]:


hw3.xval_learning_alg(learner,auto_data, auto_labels, 10)


# In[77]:


learner1=partial(hw3.averaged_perceptron, params={'T': 1})


# In[71]:


import code_for_hw3_part2 as hw3

auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# Feature set 1 — tout en raw
features_raw = [('cylinders',     hw3.raw),
                ('displacement',  hw3.raw),
                ('horsepower',    hw3.raw),
                ('weight',        hw3.raw),
                ('acceleration',  hw3.raw),
                ('origin',        hw3.raw)]

# Feature set 2 — standard + one_hot
features_std = [('cylinders',     hw3.one_hot),
                ('displacement',  hw3.standard),
                ('horsepower',    hw3.standard),
                ('weight',        hw3.standard),
                ('acceleration',  hw3.standard),
                ('origin',        hw3.one_hot)]

# Construire les data/labels pour chaque feature set
auto_data_raw, auto_labels = hw3.auto_data_and_labels(auto_data_all, features_raw)
auto_data_std, _           = hw3.auto_data_and_labels(auto_data_all, features_std)

T_values   = [1, 10, 50]
algos      = [hw3.perceptron, hw3.averaged_perceptron]
algo_names = ['perceptron', 'averaged_perceptron']
datasets   = [('raw', auto_data_raw), ('standard', auto_data_std)]

results = {}

for algo, algo_name in zip(algos, algo_names):
    for T in T_values:
        for feat_name, data in datasets:

            # xval_learning_alg attend un learner sans params
            # on wrappe pour passer T
            learner = lambda d, l, T=T, algo=algo: algo(d, l, params={'T': T})

            accuracy = hw3.xval_learning_alg(learner, data, auto_labels, k=10)

            key = f"{algo_name} | T={T} | features={feat_name}"
            results[key] = accuracy
            print(f"{key} → {accuracy:.4f}")


# In[72]:


[0.8060,0.9005]


# In[75]:


# Feature set 2 — standard + one_hot
features_std = [('cylinders',     hw3.one_hot),
                ('displacement',  hw3.standard),
                ('horsepower',    hw3.standard),
                ('weight',        hw3.standard),
                ('acceleration',  hw3.standard),
                ('origin',        hw3.one_hot)]

# Construire les data/labels pour chaque feature set
auto_data_std, _           = hw3.auto_data_and_labels(auto_data_all, features_std)
hw3.averaged_perceptron(auto_data_std,_)


# In[76]:


# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)


# In[82]:


learner1(auto_data_raw,auto_labels)
hw3.xval_learning_alg(learner1,auto_data, auto_labels, k=10)


# In[166]:


reverse_dictionary = hw3.reverse_dict(dictionary)   # déjà dans le fichier hw3 ✅
reverse_dictionary[2]


# In[233]:


th,th0=hw3.averaged_perceptron(review_bow_data,review_labels,params={'T':10})
mot_positive=[]
for i in np.argsort((th.flatten()))[:]:
    mot_positive.append(reverse_dictionary[i])
mot_positive


# In[237]:


mot_negative=[]
for i in np.argsort((th.flatten()))[:10]:
    mot_negative.append(reverse_dictionary[i])
mot_negative


# In[84]:


learner2=partial(hw3.averaged_perceptron,params={'T':10})


# In[86]:


hw3.xval_learning_alg(learner2,review_bow_data, review_labels, k=10)


# In[230]:


Y = np.array([[8,1,-4,4,5,3,2,6]])
np.argsort(Y.flatten())[:]


# In[231]:


np.argsort(Y.flatten())[:3]


# In[236]:


np.argsort(Y.flatten())[:3]


# In[33]:


mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[6]["images"]
d1 = mnist_data_all[8]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples = x.shape[0]
    return x.reshape(n_samples, -1).T 

def row_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m, n_samples) array where each entry is the average of a row
    """
    x = np.array(x)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]  # (m,n) -> (1,m,n)
    # mean over columns (axis=2), result shape: (n_samples, m), then transpose
    return x.mean(axis=2).T  # (m, n_samples)

def col_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (n, n_samples) array where each entry is the average of a column
    """
    x = np.array(x)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]  # (m,n) -> (1,m,n)
    # mean over rows (axis=1), result shape: (n_samples, n), then transpose
    return x.mean(axis=1).T  # (n, n_samples)


def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    x = np.array(x)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]  # (m,n) -> (1,m,n)
    
    m = x.shape[1]
    mid = m // 2
    top = x[:, :mid, :].mean(axis=(1, 2))
    bottom = x[:, mid:, :].mean(axis=(1, 2))
    return np.array([top, bottom])


# use this function to evaluate accuracy
x=[]
for features in [row_average_features,col_average_features,top_bottom_features] :
    acc = hw3.get_classification_accuracy(features(data), labels)
    x.append(acc)
x
#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data


# In[273]:


data=np.array([[0.2, 0.8, 0.2, 0.8],[0.2,  0.2,  0.8,  0.8],[1.,1.,1.,1.],[1,3,5,5]])
def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    aver_row=[]
    for _ in range(np.shape(x)[0]):
        aver_row.append(np.mean(x[_,:]))
    return np.reshape(aver_row,(np.shape(x)[0],1))
    
average_row(data)


# In[275]:


def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    aver_row=[]
    for _ in range(np.shape(x)[1]):
        aver_row.append(np.mean(x[:,_]))
    return np.reshape(aver_row,(np.shape(x)[1],1))
col_average_features(data)


# In[75]:


x = np.array([
    [[1, 2, 3],   # image 0, ligne 0
     [4, 5, 6]],  # image 0, ligne 1

    [[7, 8, 9],   # image 1, ligne 0
     [1, 2, 3]]   # image 1, ligne 1
])
print(np.shape(x))
x.sum(axis=1)


# In[69]:


t=np.random.rand(4, 28, 28)  # valeurs entre 0 et 1:


# In[208]:


#Week 4 Homework,margine, simply separable, Linear support vector Machins, Hinge Loss...

from numpy import linalg as la
def margine(th,th0,x,y):
    return y*(np.dot(th,x)+th0)/la.norm(th)
#th=np.array([[1,1]])  
#th0=-4
#print(point)
#data=np.array([[3,2],[1,1],[4,2]])
#y=np.array([1,-1,-1])

"""data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]])
blue_th0 = -1.5
red_th = np.array([[1, 0]])
red_th0 = -2.5"""
#marge=[]
#marge.append(margine(red_th,red_th0,data,labels).min())
#marge.append(margine(red_th,red_th0,data,labels).max())
#marge.append(np.sum(margine(red_th,red_th0,data,labels)))

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]])
th0 = -4
marge=margine(th,th0,data,labels)
print(marge <=np.sqrt(2)/2)
print(marge)
print(np.shape(marge)[1])
for i in range(np.shape(marge)[1]):
    if (marge[:,i] <= np.sqrt(2)/2):
        marge[:,i]= 1 - marge[:,i]/(np.sqrt(2)/2)
    else :
        marge[:,i]=0
marge


# In[132]:


th = np.array([[0, 48,0],[-2,29,1],[3,4,40]]).T
print(th)
th.max()


# In[192]:


1 - 0.14142136/(np.sqrt(2)/2)


# In[119]:


3/2


# In[211]:


marge


# In[306]:


#gradient descent function 
def f(x):
    return (x+1)**2
def df(x):
    return 2**(x+1)
def gd(f, df, x0, step_size_fn, max_iter):
    x=x0
    fs=[]
    xs=[]
    for i in range(max_iter):
        fs.append(f(x))
        xs.append(x)
        x = x - step_size_fn(i) * df(x)
        
    return x, np.array(fs), np.array(xs)
x=np.array([1,3])


# In[273]:


x=np.array([0.0,0.]).T
x
print(len(x))
for i in range(len(x)):
    print("hi")


# In[314]:



def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2
def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) +                (-3. + x) * (-2. + x) * (3. + x) +                (-3. + x) * (1. + x) * (3. + x) +                (-2. + x) * (1. + x) * (3. + x) +                2 * (-1. + x + y),
               2 * (-1. + x + y)])

#numerical gradiant

def num_grad(f, delta=0.001):
    def df(x):
        d = x.shape[0]
        grad = np.zeros((d, 1))
        for i in range(d):
            e = np.zeros((d, 1))
            e[i, 0] = delta
            grad[i, 0] = (f(x + e) - f(x - e)) / (2 * delta)
        return grad      
    return df
x=cv([0.,0.])
num_grad(f2,delta=0.001)(x)
def minimize(f, x0, step_size_fn, max_iter):
    
      return gd(f, num_grad(f,delta=0.001), x0, step_size_fn, max_iter)
    
minimize(f2, x, lambda i: 0.001, 5)  # ✅


# In[322]:


x=np.array([2,1])
#max(0,1-x)
np.mean(x)


# In[395]:


def hinge(v):
    return np.where(v >= 1, 0, 1 - v)


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
x,y = super_simple_separable()
th,th0 = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y*(np.dot(th.T,x)+th0))
  

hinge_loss(x, y, th, th0)
# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    return np.mean(hinge_loss(x,y,th,th0))+lam*np.linalg.norm(th)**2

svm_obj(x, y,th,th0,0.1)


# In[393]:


def hinge(v):
    return np.where(v >= 1, 0, 1 - v)

def hinge_loss(x, y, th, th0):
    return hinge(y * (np.dot(th.T, x) + th0))

def svm_obj(X, y, th, th0, lam):
    return np.mean(hinge_loss(X, y, th, th0)) + lam * np.linalg.norm(th) ** 2
svm_obj(x, y, th, th0, 0.1)


# In[396]:


y*np.array([0,1,2,1])


# In[418]:


# Returns the gradient of hinge(v) with respect to v.
#max(0,1-v)  1-v if v>=1, otherwise 0
def d_hinge(v):
    return np.where(v>=1, 0, -1)
d_hinge(y*np.array([0,1,2,1]))


# In[419]:


# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(yx)
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(np.ones(len(th0)))



# In[420]:


# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x,y,th,th0))+2*lam*np.linalg.norm(th)


# In[421]:


# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x,y,th,th0))


# In[422]:


# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(x, y, th, th0, lam):
    return np.vstack([d_svm_obj_th(x,y,th,th0,lam),d_svm_obj_th0(x,y,th,th0,lam)])


# In[429]:


# Returns the gradient of hinge(v) with respect to v.
#max(0,1-v)  1-v if v>=1, otherwise 0
def d_hinge(v):
    return np.where(v>=1, 0, -1)
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return y*x*d_hinge(y*(np.dot(th.T,x)+th0))
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return y*d_hinge(y*(np.dot(th.T,x)+ th0))
# Returns the gradient of svm_obj(x, y, th, th0) with respect to th

def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x,y,th,th0),keepdims=True,axis=1)+ 2*lam*th
    
# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x,y,th,th0),keepdims=True,axis=1)
    
# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(x, y, th, th0, lam):
    return np.vstack([d_svm_obj_th(x,y,th,th0,lam),d_svm_obj_th0(x,y,th,th0,lam)])


# In[449]:


# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return y*x*d_hinge(y*(np.dot(th.T,x)+th0)) #+ y*np.dot(th.T,x)+ d_hinge(th0)
#d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
d_hinge_loss_th(X2, y2, th2, th20).tolist()


#X2[:,0:1]*y2[:,0:1]


# In[3]:


def batch_svm_min(x, y, lam):
    
    # 1. Initialisation à zéro
    d, n  = x.shape
    th    = np.zeros((d, 1))       # theta initial = 0
    th0   = np.zeros((1, 1))       # theta0 initial = 0
    
    # 2. Point de départ — vstack theta et theta0 en un seul vecteur
    x0 = np.vstack([th, th0])      # shape (d+1, 1)
    
    # 3. Fonction objectif
    def f(tht):
        th  = tht[:-1]             # tout sauf dernière ligne = theta
        th0 = tht[-1:]             # dernière ligne = theta0
        return svm_obj(x, y, th, th0, lam)
    
    # 4. Gradient
    def df(tht):
        th  = tht[:-1]
        th0 = tht[-1:]
        return svm_obj_grad(x, y, th, th0, lam)
    
    # 5. Step size function déjà fournie
    step_size_fn = lambda i: 2 / (i + 1) ** 0.5
    
    # 6. Lance gradient descent
    return gd(f, df, x0, step_size_fn, 10)


# In[ ]:




