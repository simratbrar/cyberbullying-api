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

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

POSITIVE_WORD_COUNT = 20413
NEGATIVE_WORD_COUNT = 58464
LOG_PRIOR = -1.5912126357837744
stopwords_set = {'when', 'again', 'other', 'only', 'can', 'however', 'that', 'were', 'do', 'former', '’ll', 'before', 'therefore', 'namely', 'whither', 'thus', 'whereafter', 'yours', 'just', 'among', 'last', 'still', 'quite', '‘ll', "'s", 'cannot', 'became', 'hers', 'of', 'six', 'both', 'see', '’ve', 'because', 'whereas', 'nobody', 'her', 'anyone', 'noone', 'either', 'hereby', 'over', 'its', 'besides', 'could', 'anywhere', '‘d', 'will', 'these', 'the', 'thereafter', 'eight', 'several', 'part', 'whoever', 'become', 'at', 'a', 'becoming', 'no', 'neither', 'myself', 'put', 'seem', 'third', 'as', 'though', 'two', 'on', 'down', 'yourselves', 'i', 'afterwards', 'during', 'something', 'who', 'beyond', 'name', 'well', 'did', 'out', 'herself', 'call', 'toward', 'once', 'made', 'anyhow', 'or', 'twelve', 'none', 'mine', 'perhaps', 'this', 'we', 'seems', 'about', 'towards', 'within', 'your', 'others', 'now', 'my', 'same', 'behind', 'also', 'five', 'would', 'throughout', 'becomes', 'doing', 'meanwhile', 'very', 'thru', 'why', 'sometimes', 'really', 'is', 'formerly', 'hereupon', 'onto', 'bottom', "'re", 'keep', '’m', 'fifty', 'been', 'get', 'whereupon', 'except', 're', 'enough', 'everywhere', 'somewhere', 'hereafter', 'fifteen', '’d', 'moreover', 'via', 'what', 'him', 'back', 'even', 'upon', 'above', 'being', 'too', 'due', 'empty', 'mostly', 'while', 'between', 'itself', '‘s', 'somehow', 'to', 'was', 'may', 'twenty', '‘re', 'latter', 'where', 'along', 'it', 'more', 'but', '‘m', 'much', 'nowhere', 'less', 'which', 'then', 'four', 'might', 'three', 'any', 'often', 'make', 'should', 'you', 'nothing', 'from', 'least', 'such', 'each', 'per', 'since', 'they', 'whose', 'another', 'wherever', 'against', 'move', 'done', 'thence', 'below', 'many', 'whether', 'ca', 'our', "'m", 'seemed', 'regarding', 'hence', 'me', 'otherwise', 'if', 'here', 'already', 'them', 'into', 'she', 'whence', 'have', 'wherein', 'off', 'be', 'various', 'say', 'someone', 'sometime', 'although', 'elsewhere', 'whatever', 'in', 'has', 'with', 'whom', "'ll", '‘ve', 'had', 'together', 'own', 'his', 'ever', 'whenever', 'their', 'most', 'through', 'few', 'ours', 'amount', 'full', 'yet', 'beforehand', "'ve", 'eleven', 'ourselves', 'further', 'until', 'everything', "'d", 'latterly', 'side', 'everyone', 'indeed', 'one', 'us', 'give', 'anything', '’re', 'there', 'whole', 'ten', 'go', 'and', 'sixty', 'am', 'amongst', 'herein', 'always', 'under', 'almost', 'those', 'anyway', 'so', 'across', 'nevertheless', 'he', 'than', 'rather', 'are', 'else', 'forty', 'therein', 'without', 'around', 'himself', 'unless', 'yourself', 'next', 'top', 'how', 'thereupon', 'some', 'thereby', 'every', 'used', 'themselves', '’s', 'take', 'for', 'please', 'after', 'first', 'beside', 'does', 'serious', 'alone', 'all', 'seeming', 'show', 'an', 'using', 'by', 'never', 'hundred', 'whereby', 'up', 'front', 'nine', 'must'}

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


