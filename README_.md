# bikedemand_fastAPI

{data_folder}
  
  __init__.py: blank file
  
  data_tratamiento.py : File with the approach to calculate the missing date by interpolation method. It file allow save the new full data
  
  list_modebase.py : A base model using list to insert the data for prediction
  
  model_base.py : A base model using a dictionary for each variable for prediction
  
  values_predict.py : File to test the new full data. It extract the index from the dataset and use it in each variable 
  
  timeseries.csv and timeseries_full.csv --> datasets used in the prediction
 
{models_folder}
  
  __init__.py : blank file
  
  ran_forest.py : File used to train the model and save it in pkl format file
  
  RFregression.plk --> Random forest file

__init__.py : blank file

test_model.py : File used to test the different tests 

app.py : Application to execute the model
