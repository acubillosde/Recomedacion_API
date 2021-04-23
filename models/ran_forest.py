import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.sparse import coo_matrix

datos = pd.read_csv('data/MovieLens.csv')

def convert_ratings_df_to_matrix(
        df, shape, columns="user,item,rating".split(',')):
    data = df[columns].values
    users = data[:, 0] - 1 # correct for zero index
    items = data[:, 1] - 1 # correct for zero index
    values = data[:, 2]
    return coo_matrix((values, (users, items)), shape=shape).toarray()

n_users = datos['user'].unique().shape[0]
n_items = datos['item'].unique().shape[0]
interactions = convert_ratings_df_to_matrix(datos, shape=(n_users, n_items)).astype(np.float64)

def train_test_split(interactions, k=5, n=2):
    """Split the interactions matrix.

    It is important to remmber that it is not about remmoving rows at
    random, because it would remmove the users; instead we want to 
    remmove some of the interactions of those users with the items.

    This function calculates the minimun ratings per user in the 
    interactions matrix and take it into account to avoid removing 
    r_ui values for users with less than n*k interactions.

    Args:
        interactions (np.ndarray): contains the data to be splitted.
        k (int): this parameter should be choosen greater than the 
            precision at k that you want to compute. Think of how many 
            items you want to recommend for every user. Defaults to 5.
        n (int): number of times that a user shuld have interacted with 
            the items set so that we can move interactions from the 
            original interactions to the test set.

    Returns:
        train (np.ndarray): the training set.
        test (np.ndarray): the test set.
        
    """
    # reserve the rerutn matrices
    train = interactions.copy()
    test = np.zeros_like(train)

    # store all user indices from which we take interactions to the test set
    user_test_indices = []

    for uid in range(train.shape[0]):
        
        # get indices (item indices) of the interactions of this user
        user_interactions_indices = np.where(train[uid,:]>0)[0]
        # take k interactions only if that user has more than n*k interactions
        if len(user_interactions_indices) >= int(n*k):

            # pick k interactions to move to the test set
            valid_pick = False
            
            while valid_pick == False:
                n_attempts = 0
                temp_train = train.copy()
                test_interactions_indices = np.random.choice(user_interactions_indices, size=k, replace = False)
                # the train set should be 0 in all places the test set is non zero
                temp_train[uid, test_interactions_indices] = 0
        
                # only continue if the sampled indices for test don't leave any movies without interactions in train
                # otherwise try again
                                
                #Important note: this might lead to a infinite loop in some situations,
                #so after a certain number of failed sampling attempts we just skip the user.
                
                interactions_per_movie = np.sum(temp_train, axis = 0)
                movies_wo_interactions = np.sum( interactions_per_movie == 0 )
                if movies_wo_interactions == 0:
                    valid_pick = True
                    train = temp_train.copy()
                    del temp_train
                else:
                    n_attempts+=1
                    if n_attempts >= 1000:
                        print("Skipping user {}".format(uid))
                        continue

                    
                    
            # fill the values of the test set 
            values = interactions[uid,test_interactions_indices]
            test[uid, test_interactions_indices] = values

            user_test_indices.append(uid)
            
    return train, test

from numpy.linalg import solve

class ExplicitMF():
    def __init__(self, 
                 ratings, 
                 n_factors=40, 
                 item_reg=0.0, 
                 user_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        item_reg : (float)
            Regularization term for item latent factors
        
        user_reg : (float)
            Regularization term for user latent factors
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI), 
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))
        
        self.partial_train(n_iter)
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))
            self.user_vecs = self.als_step(self.user_vecs, 
                                           self.item_vecs, 
                                           self.ratings, 
                                           self.user_reg, 
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs, 
                                           self.user_vecs, 
                                           self.ratings, 
                                           self.item_reg, 
                                           type='item')
            ctr += 1
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter

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