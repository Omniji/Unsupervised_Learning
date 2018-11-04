# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg,nn_activation
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}/'.format(sys.argv[1])

np.random.seed(0)
digits = pd.read_hdf(out+'datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

abalone = pd.read_hdf(out+'datasets.hdf','abalone')
abaloneX = abalone.drop('Class',1).copy().values
abaloneY = abalone['Class'].copy().values


abaloneX = StandardScaler().fit_transform(abaloneX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters =  [2,5,10,15,20,25,30,35,40]

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
silhouette = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
abaloneX2D = TSNE(verbose=10, random_state=5).fit_transform(abaloneX)
digitsX2D = TSNE(verbose=10, random_state=5).fit_transform(digitsX)

for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(abaloneX)
    gmm.fit(abaloneX)
    SSE[k]['abalone'] = km.score(abaloneX)
    ll[k]['abalone'] = gmm.score(abaloneX)
    acc[k]['abalone']['Kmeans'] = cluster_acc(abaloneY,km.predict(abaloneX))
    acc[k]['abalone']['GMM'] = cluster_acc(abaloneY,gmm.predict(abaloneX))
    adjMI[k]['abalone']['Kmeans'] = ami(abaloneY,km.predict(abaloneX))
    adjMI[k]['abalone']['GMM'] = ami(abaloneY,gmm.predict(abaloneX))
    silhouette[k]['abalone']['Kmeans'] = silhouette_score(abaloneX, km.labels_, metric='euclidean')
    silhouette[k]['abalone']['GMM'] = silhouette_score(abaloneX, gmm.predict(abaloneX), metric='euclidean')

    abalone2D = pd.DataFrame(
        np.hstack((abaloneX2D, np.atleast_2d(km.predict(abaloneX)).T)), columns=['x', 'y', 'target'])
    abalone2D.to_csv(out + 'abalone2D_km_{}.csv'.format(k))
    abalone2D = pd.DataFrame(
        np.hstack((abaloneX2D, np.atleast_2d(gmm.predict(abaloneX)).T)), columns=['x', 'y', 'target'])
    abalone2D.to_csv(out + 'abalone2D_gmm_{}.csv'.format(k))

    km.fit(digitsX)
    gmm.fit(digitsX)
    SSE[k]['Digits'] = km.score(digitsX)
    ll[k]['Digits'] = gmm.score(digitsX)
    acc[k]['Digits']['Kmeans'] = cluster_acc(digitsY,km.predict(digitsX))
    acc[k]['Digits']['GMM'] = cluster_acc(digitsY,gmm.predict(digitsX))
    adjMI[k]['Digits']['Kmeans'] = ami(digitsY,km.predict(digitsX))
    adjMI[k]['Digits']['GMM'] = ami(digitsY,gmm.predict(digitsX))
    silhouette[k]['Digits']['Kmeans'] = silhouette_score(digitsX, km.labels_, metric='euclidean')
    silhouette[k]['Digits']['GMM'] = silhouette_score(digitsX, gmm.predict(digitsX), metric='euclidean')
    print(k, clock()-st)

    digits2D = pd.DataFrame(np.hstack((digitsX2D, np.atleast_2d(km.predict(digitsX)).T)), columns=['x', 'y', 'target'])
    digits2D.to_csv(out + 'digits2D_km_{}.csv'.format(k))
    digits2D = pd.DataFrame(np.hstack((digitsX2D, np.atleast_2d(gmm.predict(digitsX)).T)), columns=['x', 'y', 'target'])
    digits2D.to_csv(out + 'digits2D_gmm_{}.csv'.format(k))


SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)
silhouette = pd.Panel(silhouette)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
silhouette.ix[:,:,'Digits'].to_csv(out+' Digits silhouette.csv')
silhouette.ix[:,:,'abalone'].to_csv(out+' abelone silhouette.csv')
acc.ix[:,:,'Digits'].to_csv(out+'Digits acc.csv')
acc.ix[:,:,'abalone'].to_csv(out+'abalone acc.csv')
adjMI.ix[:,:,'Digits'].to_csv(out+'Digits adjMI.csv')
adjMI.ix[:,:,'abalone'].to_csv(out+'abalone adjMI.csv')


#%% NN fit data (2,3)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch,'NN__activation':nn_activation}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch,'NN__activation':nn_activation}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone cluster GMM.csv')




grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch,'NN__activation':nn_activation}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch,'NN__activation':nn_activation}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,scoring='f1_macro')

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster GMM.csv')

# %% For chart 4/5
abaloneX2D = TSNE(verbose=10,random_state=5).fit_transform(abaloneX)
digitsX2D = TSNE(verbose=10,random_state=5).fit_transform(digitsX)

abalone2D = pd.DataFrame(np.hstack((abaloneX2D,np.atleast_2d(abaloneY).T)),columns=['x','y','target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D,np.atleast_2d(digitsY).T)),columns=['x','y','target'])

abalone2D.to_csv(out+'abalone2D.csv')
digits2D.to_csv(out+'digits2D.csv')


