
# coding: utf-8

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, regex='\S+', remove_digits=False, english_only=False, stop_words=None, lower=True, filters=None):
        self.regex = regex
        self.remove_digits = remove_digits
        self.english_only = english_only
        self.stop_words = stop_words
        self.lower = lower
        self.filters = filters
        
    def transform(self, X, *args):
        tokenizer = RegexpTokenizer(self.regex)
        result = []
        for row in X:
            tokens = tokenizer.tokenize(row)
            if self.filters is not None:
                tokens = [re.sub(self.filters, '', t) for t in tokens]
            if self.lower:
                tokens = [t.lower() for t in tokens]
            if self.remove_digits:
                tokens = [t for t in tokens if not t.isdigit()]
            if self.english_only:
                english_words = set(nltk.corpus.words.words())
                tokens = [t for t in tokens if t in english_words]
            if self.stop_words is not None:
                tokens = [t for t in tokens if not t in self.stop_words]
            tokens = ' '.join(tokens)
            if tokens == '':
                tokens = 'cleaned'
            result.append(tokens)
        return result
    
    def fit(self, *args):
        return self

from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class KerasProcesser(BaseEstimator, TransformerMixin):
    def __init__(self, num_words, maxlen):
        self.num_words = num_words
        self.maxlen = maxlen
        
    def transform(self, X, *args):
        tokenizer = Tokenizer(self.num_words, lower=False, filters='')
        tokenizer.fit_on_texts(X)
        # vocab_size = len(tokenizer.word_index) + 1
        result = tokenizer.texts_to_sequences(X)
        result = pad_sequences(result, maxlen=self.maxlen, padding='post')
        return result, tokenizer
    
    def fit(self, *args):
        return self
