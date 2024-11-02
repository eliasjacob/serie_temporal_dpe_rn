# Time Series Forecasting API

This API provides multivariate time series forecasting capabilities using Facebook Prophet with support for exogenous variables.

## Project Structure

```
api/
├── main.py         # FastAPI application and routes
├── models.py       # Pydantic models for request/response validation
├── services.py     # Business logic for forecasting
└── config.py       # Application configuration
```

## Data Format Requirements

The input parquet file must have:
- A DatetimeIndex as the index (not a date column)
- One or more target columns for forecasting
- Optional feature columns for exogenous variables

## API Endpoints

### POST /fit
Fits the Prophet model with the provided time series data and exogenous variables.

Request format:
- `file`: Upload a parquet file containing the time series data
- `target_columns`: List of column names to be forecasted
- `feature_columns`: (Optional) List of column names to be used as features/exogenous variables

Example using curl:
```bash
curl -X POST "http://localhost:8123/fit" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.parquet" \
  -F "target_columns=[\"target1\",\"target2\"]" \
  -F "feature_columns=[\"feature1\",\"feature2\"]"
```

### POST /forecast
Makes predictions using the fitted model.

Request body:
```json
{
    "start_date": "2023-01-01T00:00:00",
    "end_date": "2023-12-31T00:00:00",
    "features_data": {
        "feature1": [1, 2, ...],
        "feature2": [1, 3, ...]
    },
    "coverage": 0.9
}
```

Response format:
```json
{
    "target1": {
        "dates": ["2023-01-01", "2023-01-02", ...],
        "predictions": [11, 11.2, ...],
        "lower_bound": [10, 10.5, ...],
        "upper_bound": [12, 12.5, ...],
        "coverage": 0.9
    },
    "target2": {
        "dates": ["2023-01-01", "2023-01-02", ...],
        "predictions": [20.1, 21.3, ...],
        "lower_bound": [19, 20, ...],
        "upper_bound": [21, 22, ...],
        "coverage": 0.9
    }
}
```

## Running the API

```bash
python main.py
```

The API will be available at http://localhost:8123. Interactive API documentation can be accessed at http://localhost:8123/docs.

## Usage Example

You can check the `usage_example.ipynb` notebook for a detailed example of how to use the API.

This will:
1. Train the model using data from api/test_data/df_train.parquet
2. Make predictions on the test period
3. Compare predictions with actual values from api/test_data/df_test.parquet
4. Display detailed results including:
   - Predicted vs actual values for each date
   - Error metrics (MAE, RMSE, MAPE)
