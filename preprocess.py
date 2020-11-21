# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 02:21:41 2020

@author: Brar
"""
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import spacy

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

POSITIVE_WORD_COUNT = 20413
NEGATIVE_WORD_COUNT = 58464
LOG_PRIOR = -1.5912126357837744
sp = spacy.load('en_core_web_sm')
stopwords_set = sp.Defaults.stop_words

def filter_token(token):
    #remove multiple continuous recurring letters
    for letter in LETTERS:
        pattern=letter+'{2,}'
        token=re.sub(pattern,letter+letter,token)
    return token


def clean_post(post):
    #change text to lowercase
    post=post.lower()
    
    #remove question tag
    post=re.sub('q:',' ',post)
    
    #remove answer tag
    post=re.sub('a:',' ',post)
    
    #remove <br> tag
    post=re.sub('<br>',' ',post)
    
    #remove &quot tag
    post=re.sub('&quot',' ',post)
    
    #filter only alphanumeric characters
    post=re.sub(r'[^a-zA-Z ]+',' ',post)
    
    words = [filter_token(word) for word in word_tokenize(post)]
    post = ' '.join(words)
    
    return post

def filter_post (post) :
    porter_stemmer = PorterStemmer()
    post = clean_post(post)
    words = [porter_stemmer.stem(word) for word in word_tokenize(post) if word not in stopwords_set]
    words = [word for word in words if len(word) > 2]
    return words

def get_likelihood(word, freq_dict, word_count):
    vocab_len = len(freq_dict)
    return np.log((freq_dict.get(word, 1) + 1) / (word_count + vocab_len))

def predict_post (post, positive_dict, negative_dict):
    words = filter_post(post)
    prob = LOG_PRIOR
    for word in words:
        prob += get_likelihood(word, positive_dict, POSITIVE_WORD_COUNT) - get_likelihood(word, negative_dict, NEGATIVE_WORD_COUNT)
    return prob > 0

def get_dict(path):
    data = pd.read_csv(path)
    freq_dict= {}
    for ind in range(len(data)):
        freq_dict[data.loc[ind]["word"]] = freq_dict.get(data.loc[ind]["word"], 0) + (int)(data.loc[ind]["frequency"])
    return freq_dict


