import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel, BaseSettings
from data.model_base import bicis
from typing import List

data = pd.read_csv('data/MovieLens.csv')

rf_pickle = open('models/RFregression.pkl', 'rb')
rf_model = pickle.load(rf_pickle)

#values
user = 5
item = 8
rating = 4


#prediction
prediction = rf_model.predict([[user, item, rating]])
print('Se recomienda:', prediction)

#Convert the dict to an array
origin = {
  "user": 1,
  "item": 6,
  "rating": 3,
}

tran_ = list(origin.values())
data_in = np.array(tran_).reshape(1,7)
# data_in = np.array(tran).reshape(1,-1)
rf_model.predict(data_in)

def predict_demand(bikes:bicis):
    df = bikes
    user = data['user']
    item =  data['item']
    rating = data['rating']

    return df

valores = [1, 6, 3]
pre = predict_demand(origin)

class bicis(BaseSettings):
    user: int
    item: int 
    rating: int


def predict_val(valores:list):
    rf_pickle = open('models/RFregression.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    # # datos = [data.data_model]
    # return np.array(data.data_model)
    prediction = rf_model.predict(valores)
    return prediction

predict_val([[5,8,4]])[0]