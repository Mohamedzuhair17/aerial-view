from functools import lru_cache
from pathlib import Path
import re

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Aerial View Image Reasoning API"

    # Model execution
    model_mode: str = "auto"  # auto | stub | real
    device: str = "cpu"
    change_weights: str = "runs/unet_train_only/checkpoints/best_changeformer_detector.pdparams"
    change_input_size: int = 512
    change_score_thresh: float = 0.25
    change_nms_iou: float = 0.5
    unet_weights: str = "runs/unet_train_only/checkpoints/unet_best.pth"
    unet_input_size: int = 256
    min_conf: float = 0.40
    min_building_area: int = 150

    # LLM
    llm_provider: str = "none"
    llm_model: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # Runtime / storage
    data_dir: Path = Path("data")
    database_backend: str = "sqlite"  # sqlite | postgres
    database_url: str = ""
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    max_upload_mb: int = 25
    max_image_side: int = 2048
    max_image_pixels: int = 12_000_000
    api_key: str = ""
    rate_limit_per_minute: int = 30
    rate_limit_max_clients: int = 10_000
    job_retention_hours: int = 72
    job_stale_hours: int = 3
    cleanup_interval_seconds: int = 3600

    # Queue: inline is convenient for local use; rq is recommended for deployments.
    job_queue_backend: str = "inline"  # inline | rq
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "aerial"

    # Legacy TorchScript ChangeFormer settings.
    change_tile_size: int = 256
    change_tile_overlap: int = 32

    # Registration
    registration_enabled: bool = True
    registration_min_matches: int = 12
    registration_min_inlier_ratio: float = 0.20
    registration_min_overlap: float = 0.60
    require_alignment: bool = False
    max_aspect_ratio_delta: float = 0.02

    # Geospatial inputs are optional; install requirements-geo.txt for GeoTIFF support.
    geospatial_enabled: bool = True
    geotiff_value_min: float = 0.0
    geotiff_value_max: float = 10000.0

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        value = value.lower()
        if value not in {"none", "gemini", "openai", "groq"}:
            raise ValueError("LLM_PROVIDER must be none, gemini, openai, or groq")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", value):
            raise ValueError("DEVICE must be cpu or cuda[:index]")
        return value

    @model_validator(mode="after")
    def validate_related_settings(self):
        if self.geotiff_value_max <= self.geotiff_value_min:
            raise ValueError("GEOTIFF_VALUE_MAX must be greater than GEOTIFF_VALUE_MIN")
        return self

    @field_validator("model_mode")
    @classmethod
    def validate_model_mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", "real"}:
            raise ValueError("MODEL_MODE must be auto or real")
        return value

    @field_validator("database_backend")
    @classmethod
    def validate_database_backend(cls, value: str) -> str:
        value = value.lower()
        if value not in {"sqlite", "postgres", "postgresql", "postgis"}:
            raise ValueError("DATABASE_BACKEND must be sqlite or postgres")
        return value

    @field_validator("job_queue_backend")
    @classmethod
    def validate_queue_backend(cls, value: str) -> str:
        value = value.lower()
        if value not in {"inline", "rq"}:
            raise ValueError("JOB_QUEUE_BACKEND must be inline or rq")
        return value

    @field_validator("min_conf", "registration_min_inlier_ratio", "registration_min_overlap", "max_aspect_ratio_delta")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("threshold settings must be between 0 and 1")
        return value

    @field_validator("max_upload_mb", "max_image_side", "max_image_pixels", "rate_limit_per_minute", "rate_limit_max_clients", "job_retention_hours", "job_stale_hours", "cleanup_interval_seconds", "registration_min_matches", "unet_input_size", "change_input_size", "min_building_area")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("this setting must be a positive integer")
        return value

    @field_validator("change_score_thresh", "change_nms_iou")
    @classmethod
    def validate_change_thresholds(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("ChangeFormer thresholds must be between 0 and 1")
        return value

    @field_validator("change_tile_overlap")
    @classmethod
    def validate_tile_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHANGE_TILE_OVERLAP cannot be negative")
        return value

    @field_validator("geotiff_value_min", "geotiff_value_max")
    @classmethod
    def validate_geotiff_values(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError("GeoTIFF stretch values must be finite")
        return value

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
