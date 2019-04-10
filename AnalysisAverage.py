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
mclDDeep = pd.read_csv('Deep1_Satisfaction.csv') #32 at 12:41
mclEDeep = pd.read_csv('Deep2_Satisfaction.csv')
Control = pd.read_csv('Control_Satisfaction.csv')


#print(mclAGradient)
plt.figure()
sns.distplot(mclAGradient,kde=False,fit_kws={"color":"darkorange"},fit=stats.gamma,color="darkorange",label="A")
sns.distplot(mclBExtreme,kde=False,fit_kws={"color":"green"},fit=stats.gamma, color="lime", label="B")
sns.distplot(mclCExtreme,kde=False,fit_kws={"color":"aqua"},fit=stats.gamma,color="aqua",label="C")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")

plt.legend()
plt.show()

plt.figure()
sns.distplot(mclDDeep,kde=False,fit_kws={"color":"navy"},fit=stats.gamma,color="navy",label="D")
sns.distplot(mclEDeep,kde=False,fit_kws={"color":"yellow"},fit=stats.gamma,color="yellow",label="E")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()

#manually ordered csv/excels so that email match up and removed those who only tasted one
mcl1 = pd.read_csv('mcl_1.csv') 
control1 = pd.read_csv('control_1.csv')
print(mcl1)
print(control1)
#array of items doing control-machine learning cookie
#we want positive values, so that machine is ranked highed than control
compare1 = mcl1['overall_sat']-control1['overall_sat']
print(compare1)
plt.figure()
sns.distplot(compare1,kde=False,fit_kws={"color":"navy"},fit=stats.gamma,color="navy",label="D")
plt.legend()
plt.show()

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

#sns.distplot(differenceA,kde=False,fit_kws={"color":"red"},fit=stats.gamma,color="red",label="A")
#sns.distplot(differenceB,kde=False,fit_kws={"color":"darkgreen"},fit=stats.gamma, color="darkgreen", label="B")
#sns.distplot(differenceC,kde=False,fit_kws={"color":"deepskyblue"},fit=stats.gamma,color="deepskyblue",label="C")
#sns.distplot(differenceD,kde=False,fit_kws={"color":"orange"},fit=stats.gamma,color="orange",label="D")
#sns.distplot(differenceE,kde=False,fit_kws={"color":"purple"},fit=stats.gamma,color="purple",label="E")
