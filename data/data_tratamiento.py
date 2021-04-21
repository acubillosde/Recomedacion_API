import datetime
import os
import numpy as np
import pandas as pd

#Interpolación para los datos de fechas faltantes
df = pd.read_csv("data/MovieLens.csv")#, parse_dates=[['dteday', 'hr']]) 

#df['dteday_hr'] = df['dteday_hr'].astype(str)
#dateparse = lambda dates: [datetime.datetime.strptime(d, '%Y-%m-%d %H') for d in dates] #:%M:%S
#df['dteday_hr'] = dateparse(df['dteday_hr'])
#df['new_time'] = [d.hour for d in df['dteday_hr']]

#df.set_index(['dteday_hr'], append = False, drop=False, inplace=True)

#df_1 = df.asfreq("H")

#df_1.mnth = df_1['mnth'].interpolate(method='linear', inplace=False)
#df_1.new_time = df_1['new_time'].interpolate(method='linear', inplace=False)
#df_1.season = df_1['season'].interpolate(method='linear', inplace=False)
#df_1.holiday = df_1['holiday'].interpolate(method='linear', inplace=False)
#df_1.weekday = df_1['weekday'].interpolate(method='linear', inplace=False)
#df_1.workingday = df_1['workingday'].interpolate(method='linear', inplace=False)
#df_1.weathersit = df_1['weathersit'].interpolate(method='linear', inplace=False)
#df_1.temp = df_1['temp'].interpolate(method='linear', inplace=False)
#df_1.atemp = df_1['atemp'].interpolate(method='linear', inplace=False)
#df_1.hum = df_1['hum'].interpolate(method='linear', inplace=False)
#df_1.windspeed = df_1['windspeed'].interpolate(method='linear', inplace=False)
#df_1.yr = df_1['yr'].interpolate(method='linear', inplace=False)
#df_1.cnt = df_1['cnt'].interpolate(method='linear', inplace=False)

#date_rng = pd.date_range(start='1/1/2011', end='01/01/2013', freq='H')
#df_fechas = df_1.reset_index(drop=True)
#fechas = pd.Series(date_rng, name='new_date')
#fechas_df = pd.concat([df_f, fechas[0:17544]], axis=1)
#fechas_df['new_date'] = fechas_df['new_date'].astype(str)
#fechas_df.to_csv('data/timeseries_full.csv')