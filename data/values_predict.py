import pandas as pd
import numpy as np


data = pd.read_csv('data/MovieLens.csv')
#new_data

#Obteniendo los valores para la predicción 
#fecha = '2011-01-01 01:00:00'

#idx = data[data['new_date'] == fecha].index[0]

user = data['user']
item =  data['item']
rating = data['rating'] 
#wheather = data.iloc[idx]['weathersit']
#temp =  data.iloc[idx]['temp']
#atemp = data.iloc[idx]['atemp']
#hum = data.iloc[idx]['hum']