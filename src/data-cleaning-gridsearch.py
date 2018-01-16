
# coding: utf-8

# Using a LSTM single model to text various cleaning steps and impact on score.
# 
# Controls:
# - CNN single model
# - maxlen: 65
# - min occurance vocab: 5
# - glove.6B.100D
# - epochs: 2
# - cv: 3
# - max features 20000

model_name = 'grid_benchmark'

import os
import logging
logging.basicConfig(filename='data_clean_search.log',level=logging.DEBUG)

dir_path = os.path.realpath('..')

# Import custom transformers

path = 'src/features'
full_path = os.path.join(dir_path, path)
import sys
sys.path.append(full_path)
from transformers import TextCleaner, KerasProcesser


# ## Import data

import numpy as np
import pandas as pd


path = 'data/raw/train.csv'

full_path = os.path.join(dir_path, path)
df_train = pd.read_csv(full_path, header=0, index_col=0)
print("Dataset has {} rows, {} columns.".format(*df_train.shape))

# fill NaN with string "unknown"
df_train.fillna('unknown',inplace=True)


# ## Pre-processing

from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed)
test_size = 0.2
target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = 'comment_text'

X = df_train[corpus][:100]
y = df_train[target][:100]


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=seed)


max_features=20000
max_length=65


# ## Model fit

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', max_features=max_features, max_length=max_length):
    model = Sequential()
    model.add(Embedding(max_features, 100, input_length=max_length))
    model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='sigmoid'))  #multi-label (k-hot encoding)
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_model(model, model_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_path + ".h5")
    print("Saved model to disk")


model = KerasClassifier(build_fn=create_model, epochs=3, verbose=2)


p = Pipeline([
    ('cleaner', TextCleaner()),
    ('keraser', KerasProcesser(num_words=max_features, maxlen=max_length)),
    ('clf', model)
])

param_grid = {"cleaner__regex": ['\S+'],
              "cleaner__remove_digits": [False],
              "cleaner__english_only": [False],
              "cleaner__stop_words": [None],
              "cleaner__filters": [r'[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n]'],
              "cleaner__lower": [True],
              "keraser__num_words": [max_features],
              "keraser__maxlen": [max_length]
             }


grid = GridSearchCV(p, param_grid=param_grid, cv=3)
grid_result = grid.fit(Xtrain, ytrain)


trained_model = grid_result.best_estimator_.named_steps['clf'].model


# save the model
model_path = os.path.join(dir_path, 'models', model_name)
save_model(trained_model, model_path)


# ## Evaluation

# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
times = grid_result.cv_results_['mean_fit_time']
for mean, stdev, param, time in zip(means, stds, params, times):
    print("%f (%f) with: %r in %f seconds" % (mean, stdev, param, time))
    logging.info("%f (%f) with: %r in %f seconds" % (mean, stdev, param, time))
    
print("Best score {} with params {}".format(grid_result.best_score_, grid_result.best_params_))
logging.info("Best score {} with params {}".format(grid_result.best_score_, grid_result.best_params_))

