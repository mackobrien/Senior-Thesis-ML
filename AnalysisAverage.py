#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:43:43 2019

@author: mackenzieobrien
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


mclAGradient = pd.read_csv('Gradient4_Satisfaction.csv')
mclBExtreme = pd.read_csv('Extreme5_Satisfaction.csv')
mclCExtreme = pd.read_csv('Extreme3_Satisfaction.csv')
#mclDDeep = pd.read_csv('Deep1_Satisfaction.csv')
#mclEDeep = pd.read_csv('Deep2_Satisfaction.csv')
#Control = pd.read_csv('Control_Satisfaction.csv')

#print(mclAGradient)
sns.distplot(mclAGradient,kde=False,fit=stats.gamma,color="maroon",label="A")
sns.distplot(mclBExtreme,kde=False,fit=stats.gamma, color="darkgreen", label="B")
sns.distplot(mclCExtreme,kde=False,fit=stats.gamma,color="deepskyblue",label="C")
plt.legend()
plt.show()
#sns.distplot(mclDDeep,kde=False,fit=stats.gamma)
#sns.distplot(mclEDeep,kde=False,fit=stats.gamma)


