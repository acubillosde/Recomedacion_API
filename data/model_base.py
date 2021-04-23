from pydantic import BaseModel, BaseSettings
from typing import List

class movies(BaseModel):
    # data_values = List[float]
    user: int
    item: int
    rating: int
