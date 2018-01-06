
# coding: utf-8

# Bag of words MLP model with Keras


import os
import numpy as np
import pandas as pd


# ## Import data

dir_path = os.path.realpath('../..')

path = 'data/interim/train.csv'

full_path = os.path.join(dir_path, path)
df_train = pd.read_csv(full_path, header=0, index_col=0)
print("Dataset has {} rows, {} columns.".format(*df_train.shape))


path = 'data/interim/test.csv'

full_path = os.path.join(dir_path, path)
df_test = pd.read_csv(full_path, header=0, index_col=0)
print("Dataset has {} rows, {} columns.".format(*df_test.shape))

# ## Encoding

from keras.preprocessing.text import Tokenizer

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokenizer(df_train.comment_text)

# encode data
Xtrain = tokenizer.texts_to_matrix(df_train.comment_text, mode='freq')
Xtest = tokenizer.texts_to_matrix(df_test.comment_text, mode='freq')


print(Xtrain.shape, Xtest.shape)


# ## Train model
from keras.models import Sequential
from keras.layers import Dense

# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # summarize defined model
    model.summary()
    return model

# define the model
n_words = Xtest.shape[1]
model = define_model(n_words)

# ## Predict model

target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = pd.DataFrame(index=df_test.index, columns=target)

for label in target:
	print('... Processing {}'.format(label))
	ytrain = df_train[label]
	# train the model
	model.fit(Xtrain, ytrain, epochs=1, verbose=2)
	# compute the predicted probabilities for X_test_dtm
	print('2.1')
	y_pred_proba = model.predict(Xtest, batch_size=1, verbose=0)
	print('2.2')
	submission[label] = y_pred_proba
	print('2.3')

path = 'data/processed/submission.csv'

full_path = os.path.join(dir_path, path)

submission.to_csv(full_path, header=True, index=True)

