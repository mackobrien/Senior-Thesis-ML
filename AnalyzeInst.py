import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import string
from num2words import num2words

dfing = pd.read_csv('ingredients.csv')
dfins = pd.read_csv('instructions.csv')

# define instructions
instructions = []
fc = {}
cnt = []
for idx, row in dfins.iterrows():
  if str(len(row['Directions'].split())) in fc:
    fc[str(len(row['Directions'].split()))] += 1
  else:
    fc[str(len(row['Directions'].split()))] = 1
  instructions.append(row['Directions'])
  cnt.append(len(row['Directions'].split()))
#print(fc)
#print(instructions)

# This plot shows that a word length of 150 contains at least 98% of the data
#plt.figure()
#plt.hist(cnt, bins=50)
#plt.show()
#
#plt.figure()
#for k,v in fc.iteritems():
#  plt.bar(int(k), v, color='r')
#plt.show()


# extract directions for recipes
def load_directions():
  mapping = dict()
  # process lines
  for idx, row in dfins.iterrows():
    line = row['Directions']
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
      print("WHAAAT!!! ---> " + line)
      continue
    recipe_id = row['Recipe Name']
    recipe_desc = tokens
    # convert direction tokens back to string
    recipe_desc = ' '.join(recipe_desc)
    # create the list if needed
    if recipe_id not in mapping:
      mapping[recipe_id] = list()
    # store direction
    mapping[recipe_id].append(recipe_desc)
  return mapping

# parse directions
directions = load_directions()
print('Loaded: %d ' % len(directions))
print(directions['Chocolate Chip Crisscross Cookies'])


def clean_directions(directions):
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for key, desc_list in directions.items():
    for i in range(len(desc_list)):
      desc = desc_list[i]
      # tokenize
      desc = desc.split()
      # convert to lower case
      desc = [word.lower() for word in desc]
      # remove punctuation from each token
      desc = [w.translate(table) for w in desc]
      # remove hanging 's' and 'a'
      #desc = [word for word in desc if len(word)>1]
      # remove tokens with numbers in them
      ndesc = []
      for word in desc:
        if word.isalpha() and len(word)>1:
          ndesc.append(word)
        elif not word.isalpha() and len(word)>0:
          ndesc.append(num2words(int(word)))
      desc = ndesc
      # store as string
      desc_list[i] =  ' '.join(desc)
 
# clean directions
clean_directions(directions)
print(directions['Chocolate Chip Crisscross Cookies'])

## https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

time.sleep(1000)
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# input layer: vector of ingredients
# Dense

# recurrent layer: produce words
# LSTM sequence length = 150
# output dimension = 50 (word vector length)





