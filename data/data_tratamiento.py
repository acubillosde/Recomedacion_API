import datetime
import os
import numpy as np
import pandas as pd

#Interpolación para los datos de fechas faltantes
df = pd.read_csv("data/timeseries.csv", parse_dates=[['dteday', 'hr']]) 

df['dteday_hr'] = df['dteday_hr'].astype(str)
dateparse = lambda dates: [datetime.datetime.strptime(d, '%Y-%m-%d %H') for d in dates] #:%M:%S
df['dteday_hr'] = dateparse(df['dteday_hr'])
df['new_time'] = [d.hour for d in df['dteday_hr']]

df.set_index(['dteday_hr'], append = False, drop=False, inplace=True)

df_4 = df.asfreq("H")

df_4.mnth = df_4['mnth'].interpolate(method='linear', inplace=False)
df_4.new_time = df_4['new_time'].interpolate(method='linear', inplace=False)
df_4.season = df_4['season'].interpolate(method='linear', inplace=False)
df_4.holiday = df_4['holiday'].interpolate(method='linear', inplace=False)
df_4.weekday = df_4['weekday'].interpolate(method='linear', inplace=False)
df_4.workingday = df_4['workingday'].interpolate(method='linear', inplace=False)
df_4.weathersit = df_4['weathersit'].interpolate(method='linear', inplace=False)
df_4.temp = df_4['temp'].interpolate(method='linear', inplace=False)
df_4.atemp = df_4['atemp'].interpolate(method='linear', inplace=False)
df_4.hum = df_4['hum'].interpolate(method='linear', inplace=False)
df_4.windspeed = df_4['windspeed'].interpolate(method='linear', inplace=False)
df_4.yr = df_4['yr'].interpolate(method='linear', inplace=False)
df_4.cnt = df_4['cnt'].interpolate(method='linear', inplace=False)
# df_4.dteday_hr = df_4['dteday_hr'].interpolate(method='linear', inplace=False)

date_rng = pd.date_range(start='1/1/2011', end='01/01/2013', freq='H')
df_f = df_4.reset_index(drop=True)
fechas = pd.Series(date_rng, name='new_date')
f_df = pd.concat([df_f, fechas[0:17544]], axis=1)
    # #Convertir en string para regresar el índice
f_df['new_date'] = f_df['new_date'].astype(str)

#Guardando el archivo 
f_df.to_csv('data/timeseries_full.csv')