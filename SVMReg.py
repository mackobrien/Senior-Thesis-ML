import os
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd

df = pd.read_csv('ingredients.csv')

print(df.head())

print(df.describe())

inCol = list(set(list(df)) - set(['Recipe Name','Rating','Calories']))




y = df['Rating'].astype(float).values
X = df[inCol].astype(float).values
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

while True:
  opt = BayesSearchCV( SVR(kernel='linear', max_iter=100000),
          { 'C': Real(1e-2, 1e+6, prior='log-uniform'), 
            'epsilon': Real(1e-2, 1e+1, prior='log-uniform'), 
          }, n_iter=100, n_points=2, n_jobs=8, cv=4, verbose=1, return_train_score=True)
  
  opt.fit(X, y)
  
  fullR = opt.best_score_
  print("cv=" + str(opt.best_score_) + " full=" + str(opt.score(X, y)))
  print(opt.best_params_)
  
  
  
  ## SVM regression
  clf = SVR(kernel='linear',
            C=opt.best_params_['C'], 
            epsilon=opt.best_params_['epsilon'],
            verbose=1)
  clf.fit(X, y)
  print(clf.score(X, y))
  
  #cc = clf.coef_
  #for x in range(len(cc)):
  #  print(str(cc[x]) + " -  " + inCol[x])
  
  
  #print(clf.intercept_ )
  
  
  print(clf.predict(scaler.transform(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))))
  print(clf.predict(scaler.transform(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))))
  
  #print(reg.predict(np.array([[3, 5]])))
  
  
  ## generating random recipe
  bestRank = -1000000
  bestRecipe = []
  
  f = 'SVMReg-res.csv'
  if os.path.exists(f):
    rdf = pd.read_csv(f, index_col=0)
  else:
    rdf = pd.DataFrame()
  
  cnt = 0
  while True:
    #new = np.abs(np.floor(np.random.normal(mu, sigma)*16)/16)
    new = np.zeros(X.shape[1])
    for x in range(X.shape[1]):
      new[x] = df[inCol[x]].sample(n=1).values
    rank = clf.predict(scaler.transform([new]))[0]
    if rank > bestRank:
      bestRank = rank
      bestRecipe = new
      bestDist = 1000000
      for recipe in X:
        dist = np.linalg.norm(recipe-new)
        if dist < bestDist:
          bestDist = dist
      print(bestRecipe)
      br = {}
      br['#Predicted Rank'] = bestRank
      br['#Uniqueness'] = bestDist
      br['#Simplicity'] = np.linalg.norm(new)
      for x in range(len(inCol)):
        br[inCol[x]] = bestRecipe[x]
      #print(br)
      rdf = rdf.append(br, ignore_index=True)
      #print(rdf)
      rdf.to_csv(f)
      print("Rank: " + str(bestRank) + " Distance: " + str(bestDist) + " Norm: " + str(br['#Simplicity']))
      cnt = 0
    cnt += 1
    if cnt > 1000000:
      break




