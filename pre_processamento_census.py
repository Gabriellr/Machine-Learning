# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:49:29 2021

@author: grodr
"""

import pandas as pd


base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder
Labelencoder_previsores = LabelEncoder()

#labels = Labelencoder_previsores.fit_transform(previsores[:,1])

previsores[:,1] = Labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = Labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = Labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = Labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = Labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = Labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = Labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = Labelencoder_previsores.fit_transform(previsores[:,13])
