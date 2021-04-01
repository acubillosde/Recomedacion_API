import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel, BaseSettings
from data.model_base import bicis
from typing import List
from data.list_modebase import values_list

fecha = '2011-01-01 01:00:00'
data = pd.read_csv('data/timeseries_full.csv')
app = FastAPI()

###Models 
#Random_Forest
rf_pickle = open('models/RFregression.pkl', 'rb')
rf_model = pickle.load(rf_pickle)

@app.get('/get_date')
async def get_date(date_time: str):
    # data = pd.DataFrame()
    idx = data[data['new_date'] == date_time].index[0]

    season = data.iloc[idx]['season']
    time =  data.iloc[idx]['new_time']
    workingday = data.iloc[idx]['workingday'] 
    wheather = data.iloc[idx]['weathersit']
    temp =  data.iloc[idx]['temp']
    atemp = data.iloc[idx]['atemp']
    hum = data.iloc[idx]['hum']

    values = {'season': season, 'time': time, 'workday':workingday,'wheather': wheather, 'temp': temp, 'atemp': atemp, 'hum':hum}
    
    return values 

@app.post('/predict')
async def predict_demand(bikes:bicis):
    rf_pickle = open('models/RFregression.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    df = bikes.dict()
    season = df['season']
    hour = df['hour']
    workingday = df['workingday']
    wheather = df['wheather']
    temp = df['temp']
    atemp = df['atemp']
    hum = df['hum']

    # return df
    get_val = list(df.values())
    # return get_val
    prediction = round(rf_model.predict([get_val])[0],3)
    result = {'The number of bikes is': prediction}
    return result

# @app.post('/test')
# async def predict_demand(valores:list):
#     rf_pickle = open('models/RFregression.pkl', 'rb')
#     rf_model = pickle.load(rf_pickle)
#     # # datos = [data.data_model]
#     # return np.array(data.data_model)
#     prediction = rf_model.predict(valores)
#     return prediction[0]


if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8000)


