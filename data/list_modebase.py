from pydantic import BaseModel, BaseSettings
from typing import List

class values_list(BaseModel):
    data_model : list[float]