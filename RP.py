

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError,nn_activation
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './RP/'
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
abalone_dims = range(1, 9)
#raise
#%% data for 1

tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(abaloneX), abaloneX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'abalone scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(digitsX), digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digits scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(abaloneX)
    tmp[dim][i] = reconstructionError(rp, abaloneX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'abalone scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(digitsX)
    tmp[dim][i] = reconstructionError(rp, digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digits scree2.csv')

#%% Data for 2

grid ={'rp__n_components':abalone_dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')

# raise

#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 8
rp = SparseRandomProjection(n_components=dim,random_state=5)

abaloneX2 = rp.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_hdf(out+'datasets.hdf','abalone',complib='blosc',complevel=9)

dim = 60
rp = SparseRandomProjection(n_components=dim,random_state=5)
digitsX2 = rp.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)