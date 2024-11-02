import pandas as pd
from sktime.forecasting.fbprophet import Prophet
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from io import BytesIO
from workalendar.america import BrazilRioGrandeDoNorte

def add_exogenous_variables(df):
    """Add exogenous variables to the dataframe."""

    input_df = df.copy()

    # Add day of week
    input_df['day_of_week'] = input_df.index.dayofweek
    
    # Add month
    input_df['month'] = input_df.index.month
    
    # Add day of the month
    input_df['day_of_month'] = input_df.index.day

    # Add holiday
    holidays = []
    cal = BrazilRioGrandeDoNorte()
    for year in range(input_df.index.min().year, input_df.index.max().year + 1):
        holidays.extend(cal.holidays(year))
    holidays = pd.to_datetime([h[0] for h in holidays])
    
    input_df['holiday'] = input_df.index.isin(holidays)

    input_df = input_df[['day_of_week', 'month', 'day_of_month', 'holiday']]

    return input_df


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
        
        # Add exogenous variables
        X = add_exogenous_variables(self.train_y)

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
        else:
            self.train_X = X

        # Fit the model
        self.model.fit(self.train_y, X=self.train_X)


    def predict(self, start_date: datetime, end_date: datetime, coverage: float = 0.9, 
                features_data: Optional[Dict[str, List[float]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Make predictions with prediction intervals and optional exogenous variables."""
        if self.model is None:
            raise ValueError("Model not fitted. Please fit the model first.")
        
        # Create forecast index
        fh = pd.date_range(
            start=start_date,
            end=end_date,
            freq='D'  # Daily frequency
        )
        
        # Prepare features data if provided
        if features_data and self.feature_columns:
            X = pd.DataFrame(features_data, index=fh)
        else:
            X = add_exogenous_variables(pd.DataFrame(index=fh))

        # Make predictions with intervals
        predictions = self.model.predict(fh, X=X)
        intervals = self.model.predict_interval(fh, X=X, coverage=coverage)
        
        # Zero out negative predictions
        predictions[predictions < 0] = 0
        intervals[intervals < 0] = 0

        return predictions, intervals
