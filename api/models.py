from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional

class TimeSeriesData(BaseModel):
    data: bytes
    target_columns: List[str]
    feature_columns: Optional[List[str]] = None  # Exogenous variables

class ForecastRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    features_data: Optional[Dict[str, List[float]]] = None  # Future values of exogenous variables

class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
