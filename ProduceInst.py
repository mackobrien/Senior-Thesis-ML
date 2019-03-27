import glob
from pickle import load
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import string
from num2words import num2words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu



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
    recipe_dirs = tokens
    # convert direction tokens back to string
    recipe_dirs = ' '.join(recipe_dirs)
    # create the list if needed
    if recipe_id not in mapping:
      mapping[recipe_id] = list()
    # store direction
    mapping[recipe_id].append(recipe_dirs)
  return mapping

# parse directions
directions = load_directions()
print('Loaded: %d ' % len(directions))
print(directions['Chocolate Chip Crisscross Cookies'])


def clean_directions(dirs):
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for key, drctns_list in dirs.items():
    for i in range(len(drctns_list)):
      drctns = drctns_list[i]
      # tokenize
      drctns = drctns.split()
      # convert to lower case
      drctns = [word.lower() for word in drctns]
      # remove punctuation from each token
      drctns = [w.translate(table) for w in drctns]
      # remove hanging 's' and 'a'
      #drctns = [word for word in drctns if len(word)>1]
      # remove tokens with numbers in them
      ndrctns = []
      for word in drctns:
        if word.isalpha() and len(word)>1:
          ndrctns.append(word)
        elif not word.isalpha() and len(word)>0:
          ndrctns.append(num2words(int(word)))
      drctns = ndrctns
      # store as string
      drctns_list[i] =  ' '.join(drctns)
    #print(key)

# clean directions
clean_directions(directions)
print(directions['Chocolate Chip Crisscross Cookies'])

# convert the loaded directions into a vocabulary of words
def to_vocabulary(directions):
  # build a list of all direction strings
  all_drctns = set()
  for key in directions.keys():
    [all_drctns.update(d.split()) for d in directions[key]]
  return all_drctns

# summarize vocabulary
vocabulary = to_vocabulary(directions)
#print(vocabulary)
print('Vocabulary Size: %d' % len(vocabulary))


# load clean directions into memory
def load_clean_directions(dirs):
  for key in dirs.keys():
    #print(dirs[key])
    # wrap direction in tokens
    drctns = 'startseq ' + dirs[key][0] + ' endseq'
    # store
    dirs[key] = drctns
  return dirs


inCol = list(set(list(dfing)) - set(['Recipe Name','Rating','Calories']))
# load recipe features
def load_recipe_features(dirs):
  features = {}
  # load all features
  for idx, row in dfing.iterrows():
    if row['Recipe Name'] in dirs.keys():
      features[row['Recipe Name']] = row[inCol].astype(float).values
    else:
      print("There was a recipe without directions --->>> " + row['Recipe Name'])
  #print(dirs.keys())
  #time.sleep(50)
  return features







# directions
train_directions = load_clean_directions(directions)
print('Descriptions: train=%d' % len(train_directions))
#print(train_directions)

# recipe features 
train_features = load_recipe_features(train_directions)
print('Recipes: train=%d' % len(train_features))
#print(train_features['Chocolate Chip Crisscross Cookies'])


# convert a dictionary of clean directions to a list of directions
def to_lines(dirs):
  #print(dirs)
  all_dir = list()
  for key in dirs.keys():
    all_dir.append(dirs[key])
    #print(all_dir)
    #time.sleep(100)
  return all_dir
 
# fit a tokenizer given caption directions
def create_tokenizer(dirs):
  lines = to_lines(dirs)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer
 
# prepare tokenizer
tokenizer = create_tokenizer(train_directions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)




# calculate the length of the direction with the most words
def max_length(directions):
  lines = to_lines(directions)
  return max(len(d.split()) for d in lines)
# determine the maximum sequence length
max_length = max_length(train_directions)
print('Description Length: %d' % max_length)


# create sequences of recipes, input sequences and output words for an recipe
def create_sequences(tokenizer, max_length, directions, recipes):
  X1, X2, y = list(), list(), list()
  # walk through each recipe identifier
  for key, drctns_list in directions.items():
    #print(key, drctns_list)
    #time.sleep(2)
    # walk through each direction for the recipe
    for drctns in [drctns_list]:
      #print(drctns)
      # encode the sequence
      seq = tokenizer.texts_to_sequences([drctns])[0]
      # split one sequence into multiple X,y pairs
      #print(seq)
      #time.sleep(2)
      for i in range(1, len(seq)):
        # split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        #print(in_seq, " -- ", out_seq)
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        #print(in_seq, " -- ", out_seq)
        # store
        #print(recipes[key])
        X1.append(recipes[key])
        X2.append(in_seq)
        y.append(out_seq)
        #print(X1)
        #print(X1[0].shape)
        #print(X2)
        #print(X2[0].shape)
        #print(y)
        #print(yshape)
        #time.sleep(20)
  return np.array(X1), np.array(X2), np.array(y)


# define the instructional model
def define_model(recipe_size, vocab_size, max_length):
  # feature extractor model
  inputs1 = Input(shape=(recipe_size,))
  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)
  # sequence model
  inputs2 = Input(shape=(max_length,))
  se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
  se2 = Dropout(0.5)(se1)
  se3 = LSTM(256)(se2)
  # decoder model
  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  # tie it together [recipe, seq] [word]
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  # summarize model
  print(model.summary())
  plot_model(model, to_file='model.png', show_shapes=True)
  return model

# map an integer to a word
def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

# generate a direction for a recipe
def generate_drctns(model, tokenizer, recipe, max_length):
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    #print([[recipe],sequence])
    #time.sleep(20)
    yhat = model.predict([[recipe],sequence], verbose=0)
    # convert probability to integer
    yhat = np.argmax(yhat)
    # map integer to word
    word = word_for_id(yhat, tokenizer)
    # stop if we cannot map the word
    if word is None:
      break
    # append as input for generating the next word
    in_text += ' ' + word
    # stop if we predict the end of the sequence
    if word == 'endseq':
      break
  #print(in_text)
  return in_text

# evaluate the skill of the model
def evaluate_model(model, directions, recipes, tokenizer, max_length):
  actual, predicted = list(), list()
  cnt = 0
  # step over the whole set
  for key, drctns_list in directions.items():
    cnt+=1
    print(cnt, "/", len(recipes))
    # generate direction
    yhat = generate_drctns(model, tokenizer, recipes[key], max_length)
    # store actual and predicted
    references = [drctns_list.split()]
    actual.append(references)
    predicted.append(yhat.split())

    #print(references)
    #print(yhat.split())

    #print('BLEU-1: %f' % sentence_bleu(references, yhat.split(), weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % sentence_bleu(references, yhat.split(), weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % sentence_bleu(references, yhat.split(), weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % sentence_bleu(references, yhat.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    #time.sleep(20)
    #if cnt > 2:
    #  break
  # calculate BLEU score
  blew_s = []
  blew_s.append(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  blew_s.append(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  blew_s.append(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  blew_s.append(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
  print('BLEU-1: %f' % blew_s[0]) 
  print('BLEU-2: %f' % blew_s[1]) 
  print('BLEU-3: %f' % blew_s[2]) 
  print('BLEU-4: %f' % blew_s[3]) 
  return blew_s
  #time.sleep(20)



X1, X2, y = create_sequences(tokenizer, max_length, train_directions, train_features)
kf = KFold(n_splits=100)
for train_index, test_index in kf.split(X1):
  X1train, X1test = X1[train_index], X1[test_index]
  X2train, X2test = X2[train_index], X2[test_index]
  ytrain, ytest = y[train_index], y[test_index]


  # define the model
  model = define_model(len(inCol), vocab_size, max_length)
  # define checkpoint callback
  filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                               save_best_only=True, mode='min')
  ### fit model
  #model.fit([X1train, X2train], ytrain, epochs=10, verbose=3, callbacks=[checkpoint], 
  #          validation_data=([X1test, X2test], ytest))
  
  
  
  
  # load the model
  list_of_files = glob.glob('model*.h5') # * means all if need specific format then *.csv
  filename = max(list_of_files, key=os.path.getctime)
  print(filename)
  model = load_model(filename)
  
  ## evaluate model
  #blew_scores = evaluate_model(model, train_directions, train_features, 
  #                             tokenizer, max_length)
  #
  #os.rename(filename, filename[:-3] + "-blew-" + str(sum(blew_scores)) + ".h5")


  for filename in os.listdir('./'):
    if filename.endswith("ChosenCookies.csv"): 
      print(filename)
      rcps_csv = pd.read_csv(filename, index_col=0)
      rcps_drctns = {}
      for idx, row in rcps_csv.iterrows():
        recipe_features = np.array(row[inCol].values)
        recipe_directions = generate_drctns(model, tokenizer, recipe_features, max_length)
        rcps_drctns[idx] = recipe_directions
        #print(recipe_directions)
      dfd = pd.DataFrame.from_dict(rcps_drctns, orient='index',
                                   columns=['Instructions'])
      dfd.to_csv(filename[:-4] + '-instr.csv')

  break


