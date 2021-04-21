import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

datos = pd.read_csv('data/MovieLens.csv')

def convert_ratings_df_to_matrix(
        df, shape, columns="user id,item id,rating".split(',')):
    data = df[columns].values
    users = data[:, 0] - 1 # correct for zero index
    items = data[:, 1] - 1 # correct for zero index
    values = data[:, 2]
    return coo_matrix((values, (users, items)), shape=shape).toarray()

n_users = datos['user id'].unique().shape[0]
n_items = datos['item id'].unique().shape[0]
interactions = convert_ratings_df_to_matrix(datos, shape=(n_users, n_items)).astype(np.float64)
#feature_columns = ["mnth", "new_time", "season", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
#target_column = "cnt"
#y_new = datos[target_column]
#X_new = datos[feature_columns]

#ind_new = datos["yr"] == 0
#X_train_n, y_train_n = X_new[ind_new], y_new[ind_new]
#X_test_n, y_test_n = X_new[~ind_new], y_new[~ind_new]

#assert X_new.shape[0] == X_train_n.shape[0] + X_test_n.shape[0]

#col = ['season','new_time','workingday',"weathersit", 'temp','atemp','hum']
#x_trainf = X_train_n[col]
#x_testf = X_test_n[col]

#model trianing 
#pipe_rf = Pipeline(steps=[("scaler", MinMaxScaler()),
#    ("rfmodel", RandomForestRegressor(n_estimators=4, max_depth=10))
#])

#pipe_rf.fit(x_trainf, y_train_n)

##Save the model
#pickle.dump(pipe_rf, open('models/RFregression.pkl', 'wb'))