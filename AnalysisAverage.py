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
sns.set(color_codes=True)

#import csvs
mclAGradient = pd.read_csv('Gradient4_Satisfaction.csv') #32
mclBExtreme = pd.read_csv('Extreme5_Satisfaction.csv') #25 answers
mclCExtreme = pd.read_csv('Extreme3_Satisfaction.csv') # 25 
#mclDDeep = pd.read_csv('Deep1_Satisfaction.csv')
#mclEDeep = pd.read_csv('Deep2_Satisfaction.csv')
#Control = pd.read_csv('Control_Satisfaction.csv')


#print(mclAGradient)
sns.distplot(mclAGradient,kde=False,fit=stats.gamma,color="red",label="A")
sns.distplot(mclBExtreme,kde=False,fit=stats.gamma, color="darkgreen", label="B")
sns.distplot(mclCExtreme,kde=False,fit=stats.gamma,color="deepskyblue",label="C")
#sns.distplot(mclDDeep,kde=False,fit=stats.gamma,color="orange",label="D")
#sns.distplot(mclEDeep,kde=False,fit=stats.gamma,color="purple",label="E")
#sns.distplot(Control,kde=False,fit=stats.gamma, color="yellow", label="Control")
plt.legend()
plt.show()



#calculate absolute 
#controlA=Control.subset(0,32)
#ControlB=Control.subset(32,57)
#ControlC=Control.subset(57,82)
#ControlD=Control.subset(83,?)
#ControlE=Control.subset(?,?)

#calculate A 
#differenceA=[];
#for i in range(0,32):
#    differenceA.append(abs(controlA[i]-mclAGradient[i]))

#calculate B 
#differenceB=[];
#for i in range(0,25):
#    differenceB.append(abs(controlB[i]-mclBExtreme[i]))

#calculate C 
#differenceC=[];
#for i in range(0,25):
#    differenceC.append(abs(controlC[i]-mclCExtreme[i]))

#calculate D 
#differenceD=[];
#for i in range(0,?):
#    differenceD.append(abs(controlD[i]-mclDDeep[i]))

#calculate E 
#differenceE=[];
#for i in range(0,?):
#    differenceE.append(abs(controlE[i]-mclEDeep[i]))