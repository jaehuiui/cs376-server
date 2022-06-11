from flask import Flask, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import math
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
MAX_SEQUENCE_LENGTH = 54


class TestDataset(Dataset) :
  #Dataset - English/typo-added/labeled
  def __init__(self, df) :
    self.df = df
  
  def __len__(self) :
    return len(self.df)
  
  def __getitem__(self, idx):
    text = self.df.iloc[idx, 0]
    item = (self.df.iloc[idx, 1], self.df.iloc[idx, 3], self.df.iloc[idx, 4])
    return text, item

app = Flask(__name__)
CORS(app)

def get_prediction(data) :
    #Tokenizer building
    total_df = pd.read_csv('./dataset.csv', sep=',')
    total_df.dropna(inplace=True)
    total_df = total_df[["text", "label"]]
    total_df["label"] = [1 if i == "nothate" else 0 for i in total_df["label"]]
    total_dataset = TestDataset(total_df)
    total_loader = DataLoader(total_dataset, batch_size=1, shuffle=True)

    texts = []
    for text, _ in total_loader:
        texts.append(text[0])
        
    tokenizer = Tokenizer(nb_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    received_text = [data]
    valid_lengths = []

    for text in received_text:
        valid_lengths.append(len(text.split(' '))) 
    sequences_test = tokenizer.texts_to_sequences(received_text)

    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))
    sequences_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding = 'post')
    model = keras.models.load_model('./model_save')

    get_fil_tgt = K.function([model.layers[0].input, model.layers[3].input],
                                    [model.layers[4].output, model.layers[5].output])

    x = [sequences_test, valid_lengths]
    [_, filter_index], preds = get_fil_tgt(x)
    result = preds.argmax(axis = -1)
    return result, filter_index

@app.route('/')
def hello():
    return 'Hello World / Heroku'

@app.route('/model')
def detection():
    data = request.args.get('data')

    if(data == ''):
        return 'hi'

    else:
        return get_prediction(data)

if __name__ == '__main__' :
    app.run(debug=True, port=5000)







