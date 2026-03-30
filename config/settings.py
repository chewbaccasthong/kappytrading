from pydantic_settings import BaseSettings
from pydantic import Field


class KalshiSettings(BaseSettings):
    api_key: str = Field(default="", alias="KALSHI_API_KEY")
    api_secret: str = Field(default="", alias="KALSHI_API_SECRET")
    base_url: str = Field(
        default="https://api.elections.kalshi.com/trade-api/v2",
        alias="KALSHI_BASE_URL",
    )


class RiskSettings(BaseSettings):
    max_position_size: int = Field(default=50, alias="MAX_POSITION_SIZE")
    max_daily_loss: int = Field(default=100, alias="MAX_DAILY_LOSS")
    max_open_positions: int = Field(default=10, alias="MAX_OPEN_POSITIONS")
    min_edge_threshold: float = Field(default=0.05, alias="MIN_EDGE_THRESHOLD")


class Settings(BaseSettings):
    kalshi: KalshiSettings = KalshiSettings()
    risk: RiskSettings = RiskSettings()
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/trades.db", alias="DATABASE_URL"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
