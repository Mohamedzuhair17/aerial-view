from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    n_buildings: Optional[int] = None
    model_mode: Optional[str] = None
    alignment: Optional[dict[str, Any]] = None


class Building(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    bbox: list[int]
    polygon: list[list[int]]
    polygons: list[list[list[int]]] = Field(default_factory=list)
    damage: str
    conf: float
    changed_ratio: float
    quadrant: str
    area: int
    geometry_geojson: Optional[dict[str, Any]] = None


class ChatIn(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=2, max_length=500)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("question must contain at least two non-whitespace characters")
        return value


class ChatOut(BaseModel):
    answer: str
    tool: str
    args: dict[str, Any]
    highlight_ids: list[str]
    evidence: dict[str, Any]
    reasoning_mode: str
    overlay_url: Optional[str] = None
