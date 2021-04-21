from pydantic import BaseModel, BaseSettings
from typing import List

class movies(BaseModel):
    # data_values = List[float]
    user: int
    item: int
    rating: int
    #wheather: int
    #temp: float
    #atemp: float 
    #hum: float

# class values_list(BaseModel):
#     data_model = list[float]