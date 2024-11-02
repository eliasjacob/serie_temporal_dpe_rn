import pandas as pd
from sktime.forecasting.fbprophet import Prophet
from typing import List, Dict, Optional
from datetime import datetime
from io import BytesIO

class ForecastingService:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def fit(self, binary_data: bytes, target_columns: List[str], feature_columns: Optional[List[str]] = None) -> None:
        """Fit the Prophet model with optional exogenous variables.
        
        Args:
            binary_data: Binary content of a parquet file
            target_column: Name of the target column
            feature_columns: List of feature column names
        """
        # Read the parquet file from binary data
        df = pd.read_parquet(BytesIO(binary_data))
        
        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Prepare target variable
        self.train_y = df[target_columns]
        
        # Initialize Prophet model
        self.model = Prophet(
                seasonality_mode="additive",
                add_country_holidays={"country_name": "Brazil"},
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
                verbose=0
            )

        
        # Add exogenous variables if provided
        print(feature_columns)
        if feature_columns:
            self.feature_columns = feature_columns
            self.train_X = df[feature_columns]
        
        # Fit the model
        if feature_columns:
            self.model.fit(self.train_y, X=self.train_X)
        else:
            self.model.fit(self.train_y)

    def predict(self, start_date: datetime, end_date: datetime, features_data: Optional[Dict[str, List[float]]] = None) -> pd.Series:
        """Make predictions with optional exogenous variables."""
        if self.model is None:
            raise ValueError("Model not fitted. Please fit the model first.")
        
        # Create forecast index
        fh = pd.date_range(
            start=start_date,
            end=end_date,
            freq='D'  # Daily frequency
        )
        
        # Prepare features data if provided
        X = None
        if features_data and self.feature_columns:
            X = pd.DataFrame(features_data, index=fh)
        
        # Make predictions
        return self.model.predict(fh, X=X)
