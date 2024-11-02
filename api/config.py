from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_FREQUENCY: str = "D"  # Daily frequency
    HOST: str = "0.0.0.0"
    PORT: int = 8123
    
settings = Settings()
