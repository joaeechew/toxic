
# coding: utf-8

# In[6]:


import os
import pandas as pd


# In[7]:


dir_path = os.path.realpath('..')


# In[8]:


path = 'data/raw/train.csv'

full_path = os.path.join(dir_path, path)
df_train = pd.read_csv(full_path, header=0, index_col=0)
print("Dataset has {} rows, {} columns.".format(*df_train.shape))


# In[9]:


path = 'data/raw/test.csv'

full_path = os.path.join(dir_path, path)
df_test = pd.read_csv(full_path, header=0, index_col=0)
print("Dataset has {} rows, {} columns.".format(*df_test.shape))


# ## Data cleaning

# In[10]:


# fill NaN with string "unknown"
df_train.fillna('unknown',inplace=True)
df_test.fillna('unknown',inplace=True)


# ## Clean text

# In[234]:


from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import RegexpTokenizer

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, regex=r'[\w]+', remove_digits=False, english_only=False, stop_words=None):
        self.regex = regex
        self.remove_digits = remove_digits
        self.english_only = english_only
        self.stop_words = stop_words
        
    def transform(self, X, *args):
        tokenizer = RegexpTokenizer(self.regex)
        result = []
        for row in X:
            tokens = tokenizer.tokenize(row)
            if self.remove_digits:
                tokens = [t for t in tokens if not t.isdigit()]
            if self.english_only:
                english_words = set(nltk.corpus.words.words())
                tokens = [t for t in tokens if t in english_words]
            if self.stop_words is not None:
                tokens = [t for t in tokens if not t in self.stop_words]
            tokens = ' '.join(tokens)
            result.append(tokens)
        return result
    
    def fit(self, *args):
        return self


# In[235]:


from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class KerasProcesser(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def transform(self, X, *args):
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(X)
        # vocab_size = len(tokenizer.word_index) + 1
        result = tokenizer.texts_to_sequences(X)
        result = pad_sequences(result, maxlen=5, padding='post')
        return result
    
    def fit(self, *args):
        return self


# In[236]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', vocab_size=vocab_size, max_length=max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='sigmoid'))  #multi-label (k-hot encoding)
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[237]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[238]:


test_size = 0.2
seed = 42
target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df_train.comment_text.values[:10]
y = df_train[target][:10]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[239]:


vocab_size=b
max_length=5


# In[240]:


model = KerasClassifier(build_fn=create_model, epochs=2, verbose=1)


# In[241]:


p = Pipeline([
    ('cleaner', TextCleaner()),
    ('keraser', KerasProcesser()),
    ('clf', model)
])


# In[242]:


param_grid = {"cleaner__regex": [r'[\w]+', r'[\s]'],
              "cleaner__remove_digits": [True, False]
             }


# In[243]:


grid = GridSearchCV(p, param_grid=param_grid, cv=2)
grid_result = grid.fit(X, y)


# In[244]:


# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

