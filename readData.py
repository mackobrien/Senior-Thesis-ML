import pandas as pd
from sklearn.svm import SVR
import numpy as np

df = pd.read_csv('ingredients.csv')

print df.head()

print df.describe()

inCol = ['Calories','Sugar (c)']

y = df['Rating'].astype(float).as_matrix()
X = df[inCol].astype(float).as_matrix()

print X
print y

print X.shape
print y.shape

clf = SVR(verbose=1)
print clf
clf.fit(X, y) 

print "predictions:"
print clf.predict(y)

