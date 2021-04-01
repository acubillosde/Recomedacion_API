from pydantic import BaseModel, BaseSettings
from typing import List

class bicis(BaseModel):
    # data_values = List[float]
    season: int
    hour: int 
    workingday: int
    wheather: int
    temp: float
    atemp: float 
    hum: float

# class values_list(BaseModel):
#     data_model = list[float]