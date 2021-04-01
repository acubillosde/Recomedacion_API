import pandas as pd
import numpy as np


data = pd.read_csv('data/timeseries_full.csv')
#new_data

#Obteniendo los valores para la predicción 
fecha = '2011-01-01 01:00:00'

idx = data[data['new_date'] == fecha].index[0]

season = data.iloc[idx]['season']
time =  data.iloc[idx]['new_time']
workingday = data.iloc[idx]['workingday'] 
wheather = data.iloc[idx]['weathersit']
temp =  data.iloc[idx]['temp']
atemp = data.iloc[idx]['atemp']
hum = data.iloc[idx]['hum']