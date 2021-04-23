import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel, BaseSettings
from data.model_base import bicis
from typing import List
from data.list_modebase import values_list

data = pd.read_csv('data/MovieLens.csv')
app = FastAPI()

###Models 
#Random_Forest
rf_pickle = open('models/RFregression.pkl', 'rb')
rf_model = pickle.load(rf_pickle)

@app.get('/get_date')
async def get_date(date_time: str):
    # data = pd.DataFrame()
    #idx = data[data['new_date'] == date_time].index[0]

    user = data['user']
    item =  data['item']
    rating = data['rating']

    values = {'user': user, 'item': item, 'rating':rating}
    
    return values 

@app.post('/predict')
async def predict_demand(bikes:bicis):
    rf_pickle = open('models/RFregression.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    df = bikes.dict()
    user = data['user']
    item =  data['item']
    rating = data['rating']

    get_val = list(df.values())
    
    prediction = round(rf_model.predict([get_val])[0],3)
    result = {'The rating is': prediction}
    return result

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8000)


