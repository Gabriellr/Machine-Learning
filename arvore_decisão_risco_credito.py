# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 08:48:10 2021

@author: grodr
"""

import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
print(classificador.feature_importances_)


from sklearn.datasets import load_iris
from sklearn import tree

tree.export_graphviz(classificador,
                       out_file ='arvore.dot',
                       feature_names=['historia','divida','garantes','rendas'],
                       class_names=['altos','moderado','baixo'],
                       filled = True,
                       leaves_parallel=True)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)