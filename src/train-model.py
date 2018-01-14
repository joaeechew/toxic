# coding: utf-8

import os
import numpy as np
import pandas as pd
import pickle
from numpy import asarray
from numpy import zeros

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# ## Settings
model_name = 'glove_model_'
seed = 42
np.random.seed(seed)

# ## Import data

dir_path = os.path.realpath('..')

path = 'data/processed/train.csv'

full_path = os.path.join(dir_path, path)
df = pd.read_csv(full_path, header=0, index_col=0)
print("Training set has {} rows, {} columns.".format(*df.shape))


# ## Train test split
test_size = 0.2
target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df.drop(target, axis=1)
y = df[target]
corpus = 'comment_text'

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=seed)


# ## Pre-processing

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(Xtrain[corpus].astype(str))

#define vocab size and max len
vocab_size = len(t.word_index) + 1
max_length = max([len(s.split()) for s in Xtrain[corpus]])
print('Vocabulary size: %d' % vocab_size)
print('Maximum length: %d' % max_length)

# integer encode the documents
encoded_Xtrain = t.texts_to_sequences(Xtrain[corpus].astype(str))
encoded_Xtest = t.texts_to_sequences(Xtest[corpus].astype(str))

# pad documents
padded_train = pad_sequences(encoded_Xtrain, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_Xtest, maxlen=max_length, padding='post')

# load the embedding into memory
embeddings_index = dict()
f = open('/Users/joaeechew/dev/glove.6B/glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)\

# ## Model fit

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', vocab_size=vocab_size, max_length=max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
#   model.summary()
#   plot_model(model, to_file='model.png', show_shapes=True)
    return model

def save_model(model, model_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_path + ".h5")
    print("Saved model to disk")

model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=10, verbose=1)

# Tuning the model
param_grid = { "clf__optimizer": ['Adam']
             }

# Notes:
# - Important parameters: kernel size, no. of feature maps
# - 1-max pooling generally outperforms otehr types of pooling
# - Dropout has little effect
# - Gridsearch across kernel size in the range 1-10
# - Search no. of filters from 100-600 and dropout of 0.0-0.5
# - Explore tanh, relu, linear activation functions

# Define pipeline
pipeline = Pipeline([
    ('clf', model)
])

# fit the model

for label in target:
    print('... Processing {}'.format(label))
    y = ytrain[label]
    # train the model
    grid = GridSearchCV(pipeline, param_grid=param_grid, verbose=1, cv=2)
    grid_result = grid.fit(padded_train, y)
    # summarize results
    print("Best {} : {} using {}".format(label, grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # save the model
    trained_model = grid_result.best_estimator_.named_steps['clf'].model
    model_path = os.path.join(dir_path, 'models', model_name+label)
    save_model(trained_model, model_path)

# ## Evaluation

from sklearn.metrics import log_loss


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


y_pred = pd.DataFrame(index=ytest.index, columns=target)
scores =[]
for label in target:
    print('... Processing {}'.format(label))
    model_path = os.path.join(dir_path, 'models', model_name+label)

    # load the model
    loaded_model = load_model(model_path)

    # evaluate model on test dataset
    y_pred[label] = loaded_model.predict(padded_test, verbose=1, batch_size=1)
    loss = log_loss(ytest[label], y_pred[label])
    scores.append(loss)
    print("Log loss for {} is {} .".format(label, loss))
    print("Combined log loss is {} .".format(np.mean(scores)))




