# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:00:21 2021

@author: Gabriel 
"""


import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()