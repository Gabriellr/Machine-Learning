# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:00:21 2021

@author: Gabriel 
"""


import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age']<0]
#apagar a somente registros com problemas
base.drop('age',1,inplace=True)
base.describe()
base.drop(base[base.age < 0].index, inplace=True)
#Preencher os valores manualmente
#Preencher os valores das medias
base.mean()
base['age'].mean()
base['age'][base.age >0].mean()
base.loc[base.age < 0, 'age'] =40.92
base.describe()

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]


previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
Imputer = Imputer(missing_values ='NaN', strategy='mean', axis=0)
imputerr = Imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = Imputer.transform(previsores[:,0:3])