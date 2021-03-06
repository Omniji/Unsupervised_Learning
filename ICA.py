

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, nn_activation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './ICA/'

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

ica = FastICA(random_state=5)
kurt = {}
for dim in abalone_dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(abaloneX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'abalone scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(digitsX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'digits scree.csv')
# raise

#%% Data for 2

grid ={'ica__n_components':abalone_dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
ica = FastICA(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'NN__activation':nn_activation}
ica = FastICA(random_state=5)
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')

#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 4
ica = FastICA(n_components=dim,random_state=10)

abaloneX2 = ica.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_hdf(out+'datasets.hdf','abalone',complib='blosc',complevel=9)

dim = 40
ica = FastICA(n_components=dim,random_state=10)
digitsX2 = ica.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)