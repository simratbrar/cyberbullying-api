# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:02:11 2020

@author: Brar
"""

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from keras.models import load_model
from keras.preprocessing import text, sequence
import pandas as pd
from preprocess import predict_post, get_dict, clean_post


app = Flask(__name__)
api = Api(app)

MAX_FEATURES = 15000
MAXLEN = 100

def initialize_lstm():
    model = load_model('./data/cyberbullying-model-categorical.h5')

    train = pd.read_csv("./data/train_text.csv")
    list_sentences_train = train["posts"].values
    tokenizer = text.Tokenizer(num_words = MAX_FEATURES)
    tokenizer.fit_on_texts(list(list_sentences_train))
    return tokenizer, model

tokenizer, model = initialize_lstm()

def initialize_naive_bayes():
    POSITIVE_DICT_PATH = './data/positive_dict.csv'
    NEGATIVE_DICT_PATH = './data/negative_dict.csv'
    positive_dict = get_dict(POSITIVE_DICT_PATH)
    negative_dict = get_dict(NEGATIVE_DICT_PATH)
    return positive_dict, negative_dict

positive_freq_dict, negative_freq_dict = initialize_naive_bayes()
    
class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        post = clean_post(posted_data['text'])
        list_sentences_test = [post]
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
        input_sequence = sequence.pad_sequences(list_tokenized_test, maxlen = MAXLEN)
        prediction = model.predict(input_sequence)
        
        toxic = prediction[0][1] > 0.7
        toxic = toxic or predict_post(post, positive_freq_dict, negative_freq_dict)
        toxic = (int)(toxic)
        response = jsonify({
                            'Toxic' : toxic
                            })
        return response

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(threaded = True)
        