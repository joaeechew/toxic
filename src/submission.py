# coding: utf-8

import os
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

# ## Settings
model_name = 'glove_model_'
seed = 42
np.random.seed(seed)
corpus = 'comment_text'

# ## Import data

dir_path = os.path.realpath('..')

path = 'data/processed/test.csv'

full_path = os.path.join(dir_path, path)
df = pd.read_csv(full_path, header=0, index_col=0)
print("Test set has {} rows, {} columns.".format(*df.shape))


# ## Pre-processing

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)

vocab_size = len(t.word_index) + 1
max_length = 1000

# integer encode and pad the documents
encoded_test = t.texts_to_sequences(df[corpus].astype(str))
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')


# ## Predict and submit

def load_model(model_path):
    # load json and create model
    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+".h5")
    print("Loaded model from disk")
    return loaded_model

target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = pd.DataFrame(index=df.index, columns=target)

for label in target:
    print('... Processing {}'.format(label))

    # load the model
    model_path = os.path.join(dir_path, 'models', model_name+label)
    model = load_model(model_path)

    y_pred_proba = model.predict(padded_test, verbose=2, batch_size=1)
    submission[label] = y_pred_proba.flatten()

## Output submissions

path = 'data/submissions/' + model_name + '.csv'

dir_path = os.path.realpath('..')
full_path = os.path.join(dir_path, path)

submission.to_csv(full_path, header=True, index=True)



