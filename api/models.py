from pydantic import BaseModel, Field
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
    coverage: float = Field(default=0.9, ge=0.0, le=1.0)  # Confidence interval coverage, between 0 and 1

class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    coverage: float