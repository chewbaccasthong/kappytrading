from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Kalshi API
    kalshi_api_key: str = Field(default="")
    kalshi_private_key_path: str = Field(default="kalshi_private_key.pem")
    kalshi_base_url: str = Field(default="https://demo-api.kalshi.co/trade-api/v2")
    kalshi_demo: bool = Field(default=True)

    # Risk Management
    max_position_size: int = Field(default=50)
    max_daily_loss: int = Field(default=100)
    max_open_positions: int = Field(default=10)
    min_edge_threshold: float = Field(default=0.05)

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///data/trades.db")

    # Logging
    log_level: str = Field(default="INFO")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
