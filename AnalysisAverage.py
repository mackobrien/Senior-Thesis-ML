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
mclAGradient = pd.read_csv('Gradient4_Satisfaction.csv',header=None) #32
mclBExtreme = pd.read_csv('Extreme5_Satisfaction.csv',header=None) #25 answers
mclCExtreme = pd.read_csv('Extreme3_Satisfaction.csv',header=None) # 25 
mclDDeep = pd.read_csv('Deep1_Satisfaction.csv',header=None) #32 at 12:41
mclEDeep = pd.read_csv('Deep2_Satisfaction.csv',header=None)
Control = pd.read_csv('Control_Satisfaction.csv',header=None)


#print(mclAGradient)
plt.figure()
sns.distplot(mclAGradient,kde=False,fit_kws={"color":"darkorange"},fit=stats.gamma,color="darkorange",label="A")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")

plt.legend()
plt.show()

plt.figure()
sns.distplot(mclBExtreme,kde=False,fit_kws={"color":"green"},fit=stats.gamma, color="lime", label="B")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()


plt.figure()
sns.distplot(mclCExtreme,kde=False,fit_kws={"color":"aqua"},fit=stats.gamma,color="aqua",label="C")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()

plt.figure()
#plt.xlim(0,10)

sns.distplot(mclDDeep,kde=False,fit_kws={"color":"navy"},fit=stats.gamma,color="navy",label="D")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()

plt.figure()
sns.distplot(mclEDeep,kde=False,fit_kws={"color":"yellow"},fit=stats.gamma,color="yellow",label="E")
sns.distplot(Control,kde=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()
#print(mclAGradient)

plt.figure()
sns.distplot(mclAGradient,kde=False,hist=False,fit_kws={"color":"darkorange"},fit=stats.gamma,color="darkorange",label="A")
sns.distplot(mclBExtreme,kde=False, hist=False,fit_kws={"color":"green"},fit=stats.gamma, color="lime", label="B")
sns.distplot(mclCExtreme,kde=False,hist=False,fit_kws={"color":"aqua"},fit=stats.gamma,color="aqua",label="C")
sns.distplot(mclDDeep,kde=False,hist=False,fit_kws={"color":"navy"},fit=stats.gamma,color="navy",label="D")
sns.distplot(mclEDeep,kde=False,hist=False,fit_kws={"color":"yellow"},fit=stats.gamma,color="yellow",label="E")
sns.distplot(Control,kde=False,hist=False,fit_kws={"color":"magenta"},fit=stats.gamma, color="magenta", label="Control")
plt.legend()
plt.show()
#Batch A
#manually ordered csv/excels so that email match up and removed those who only tasted one
mcl1 = pd.read_csv('mcl_1.csv') 
control1 = pd.read_csv('control_1.csv')

#array of items doing control-machine learning cookie
#we want positive values, so that machine is ranked highed than control
compare1 = mcl1['overall_sat']-control1['overall_sat']
#print(compare1)


#Batch B
mcl2 = pd.read_csv('mlc_B.csv') 
control2 = pd.read_csv('batch2.csv')
compare2 = mcl2['overall_sat']-control2['overall_sat']
#print(compare5)

#Batch C
mcl3 = pd.read_csv('mlc_C.csv') 
control3 = pd.read_csv('batch3.csv')
compare3 = mcl3['overall_sat']-control3['overall_sat']


#Batch D
mcl4 = pd.read_csv('mlc_D.csv') 
control4 = pd.read_csv('control4.csv')
compare4 = mcl4['overall_sat']-control4['overall_sat']

#Batch E
mcl5 = pd.read_csv('mlc5.csv') 
control5 = pd.read_csv('control5.csv')
compare5 = mcl5['overall_sat']-control5['overall_sat']

#display plot of comparison
plt.figure()
sns.distplot(compare1,kde=False,hist=False,fit_kws={"color":"navy"},fit=stats.gamma,color="navy",label="Batch A")
sns.distplot(compare2,kde=False,hist=False,fit_kws={"color":"lime"},fit=stats.gamma,color="lime",label=" Batch B")
sns.distplot(compare3,kde=False,hist=False,fit_kws={"color":"magenta"},fit=stats.gamma,color="magenta",label=" Batch C")
sns.distplot(compare4,kde=False,hist=False,fit_kws={"color":"orange"},fit=stats.gamma,color="orange",label=" Batch D")
sns.distplot(compare5,kde=False,hist=False,fit_kws={"color":"blue"},fit=stats.gamma,color="blue",label=" Batch E")

plt.legend()
plt.show()
