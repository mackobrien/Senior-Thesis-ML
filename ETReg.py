import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

df = pd.read_csv('ingredients.csv')

print(df.head())

print(df.describe())

inCol = list(set(list(df)) - set(['Recipe Name','Rating','Calories']))



y = df['Rating'].astype(float).values
X = df[inCol].astype(float).values

while True:
  ## linear regression
  reg = ExtraTreesRegressor().fit(X, y)
  print(reg.score(X, y))
  
  cc = reg.feature_importances_
  for x in range(len(cc)):
    print(str(cc[x]) + " -  " + inCol[x])
  
  
  print(reg.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  print(reg.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  
  
  
  ## generating random recipe
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  bestRank = -1000000
  bestRecipe = []
  
  f = 'ETReg-res.csv'
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
    rank = reg.predict([new])
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
      br['#Predicted Rank'] = bestRank[0]
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
  



