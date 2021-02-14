# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:15:35 2021

@author: grodr
"""

import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit(previsores[:,0])
previsores[:,0] = labelencoder.fit(previsores[:,0])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)