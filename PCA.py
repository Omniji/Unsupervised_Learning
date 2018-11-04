# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg,nn_activation
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

abalone = pd.read_hdf('./BASE/datasets.hdf','abalone')        
abaloneX = abalone.drop('Class',1).copy().values
abaloneY = abalone['Class'].copy().values


abaloneX = StandardScaler().fit_transform(abaloneX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
abalone_dims = range(1,9)
#raise
#%% data for 1

pca = PCA(random_state=5)
pca.fit(abaloneX)
tmp = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,9))
tmp.to_csv(out+'abalone scree.csv')


pca = PCA(random_state=5)
pca.fit(digitsX)
tmp = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,65))
tmp.to_csv(out+'digits scree.csv')

#%% Data for 2

grid ={'pca__n_components':abalone_dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
pca = PCA(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
pca = PCA(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')

#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 6
pca = PCA(n_components=dim,random_state=10)

abaloneX2 = pca.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_hdf(out+'datasets.hdf','abalone',complib='blosc',complevel=9)

dim = 40
pca = PCA(n_components=dim,random_state=10)
digitsX2 = pca.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)