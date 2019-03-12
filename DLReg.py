import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv('ingredients.csv')

print(df.head())

print(df.describe())

inCol = list(set(list(df)) - set(['Recipe Name','Rating','Calories']))

y = df['Rating'].astype(float).values
X = df[inCol].astype(float).values

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define base model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(len(inCol), input_dim=len(inCol), kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# define the model
def larger_model():
  # create model
  model = Sequential()
  model.add(Dense(128, input_dim=len(inCol), kernel_initializer='normal', activation='relu'))
  model.add(Dense(32, kernel_initializer='normal', activation='relu'))
  model.add(Dense(8, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# define wider model
def wider_model():
  # create model
  model = Sequential()
  model.add(Dense(256, input_dim=len(inCol), kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

while True:
  # evaluate model with standardized dataset
  estimator = KerasRegressor(build_fn=baseline_model, epochs=400, batch_size=5,
                             verbose=0)
  
  kfold = KFold(n_splits=10, random_state=seed)
  results = cross_val_score(estimator, X, y, cv=kfold, n_jobs=8)
  print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
  
  estimator.fit(X, y)
  prediction = estimator.predict(X)
  print(r2_score(y, prediction))
  
  print(estimator.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  print(estimator.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  
  ## generating random recipe
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  bestRank = -1000000
  bestRecipe = []
  
  f = 'NNReg-res.csv'
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
    rank = estimator.predict(np.array([new]))
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
  cnt = 0
  
  
  
  
  # evaluate model with standardized dataset
  np.random.seed(seed)
  estimators = []
  estimators.append(('standardize', StandardScaler()))
  estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=400,
                                           batch_size=5, verbose=0)))
  pipeline = Pipeline(estimators)
  kfold = KFold(n_splits=10, random_state=seed)
  results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=8)
  print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
  
  pipeline.fit(X, y)
  prediction = pipeline.predict(X)
  print(r2_score(y, prediction))
  
  print(pipeline.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  print(pipeline.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  
  ## generating random recipe
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  bestRank = -1000000
  bestRecipe = []
  
  f = 'NNNReg-res.csv'
  if os.path.exists(f):
    rdf = pd.read_csv(f, index_col=0)
  else:
    rdf = pd.DataFrame()
  
  while True:
    #new = np.abs(np.floor(np.random.normal(mu, sigma)*16)/16)
    new = np.zeros(X.shape[1])
    for x in range(X.shape[1]):
      new[x] = df[inCol[x]].sample(n=1).values
    rank = pipeline.predict(np.array([new]))
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
  cnt = 0
  
  
  
  np.random.seed(seed)
  estimators = []
  estimators.append(('standardize', StandardScaler()))
  estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=400, 
                                           batch_size=5, verbose=0)))
  pipeline = Pipeline(estimators)
  kfold = KFold(n_splits=10, random_state=seed)
  results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=8)
  print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
  
  
  pipeline.fit(X, y)
  prediction = pipeline.predict(X)
  print(r2_score(y, prediction))
  
  print(pipeline.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  print(pipeline.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  
  ## generating random recipe
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  bestRank = -1000000
  bestRecipe = []
  
  f = 'DLReg-res.csv'
  if os.path.exists(f):
    rdf = pd.read_csv(f, index_col=0)
  else:
    rdf = pd.DataFrame()
  
  while True:
    #new = np.abs(np.floor(np.random.normal(mu, sigma)*16)/16)
    new = np.zeros(X.shape[1])
    for x in range(X.shape[1]):
      new[x] = df[inCol[x]].sample(n=1).values
    rank = pipeline.predict(np.array([new]))
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
  cnt = 0
  
  
  
  
  
  np.random.seed(seed)
  estimators = []
  estimators.append(('standardize', StandardScaler()))
  estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=400, 
                                           batch_size=5, verbose=0)))
  pipeline = Pipeline(estimators)
  kfold = KFold(n_splits=10, random_state=seed)
  results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=8)
  print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
  
  
  pipeline.fit(X, y)
  prediction = pipeline.predict(X)
  print(r2_score(y, prediction))
  
  print(pipeline.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  print(pipeline.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
  
  ## generating random recipe
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  bestRank = -1000000
  bestRecipe = []
  
  f = 'WNNReg-res.csv'
  if os.path.exists(f):
    rdf = pd.read_csv(f, index_col=0)
  else:
    rdf = pd.DataFrame()
  
  rdf = pd.DataFrame()
  while True:
    #new = np.abs(np.floor(np.random.normal(mu, sigma)*16)/16)
    new = np.zeros(X.shape[1])
    for x in range(X.shape[1]):
      new[x] = df[inCol[x]].sample(n=1).values
    rank = pipeline.predict(np.array([new]))
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
  cnt = 0




