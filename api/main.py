from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from typing import List, Dict
import pandas as pd
import traceback
from models import TimeSeriesData, ForecastRequest, ForecastResponse
from services import ForecastingService
from config import settings

app = FastAPI(title="Time Series Forecasting API")

# Initialize the forecasting service
forecasting_service = ForecastingService()

@app.post("/fit")
async def fit_model(
    file: UploadFile = File(...),
    target_columns: List[str] = Form(...),
    feature_columns: List[str] = Form([])  # Changed to default empty list
):
    try:
        # Read the binary content of the uploaded file
        binary_data = await file.read()
        
        # Print debug information
        print(f"Received target_column: {target_columns}")
        print(f"Received feature_columns: {feature_columns}")
        
        forecasting_service.fit(
            binary_data=binary_data,
            target_columns=target_columns,
            feature_columns=feature_columns
        )
        return {"message": "Model fitted successfully"}
        
    except ValueError as e:
        print(f"ValueError in fit_model: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Exception in fit_model: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=Dict[str, ForecastResponse])
async def make_forecast(request: ForecastRequest):
    try:
        predictions_df = forecasting_service.predict(
            start_date=request.start_date,
            end_date=request.end_date,
            features_data=request.features_data
        )
        
        # Create forecast index
        fh = pd.date_range(
            start=request.start_date,
            end=request.end_date,
            freq=settings.MODEL_FREQUENCY
        )
        
        # Create response with predictions for each target
        response = {}
        for column in predictions_df.columns:
            response[column] = ForecastResponse(
                dates=[d.strftime("%Y-%m-%d") for d in fh],
                predictions=predictions_df[column].tolist()
            )
        
        return response
        
    except ValueError as e:
        print(f"ValueError in make_forecast: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Exception in make_forecast: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Pass the application as an import string
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
