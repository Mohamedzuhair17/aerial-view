"""Safe filesystem paths for generated analysis jobs."""
import re
from pathlib import Path

from app.config import settings

_JOB_ID = re.compile(r"^[0-9a-f]{12,32}$")


def job_directory(job_id: str) -> Path:
    value = str(job_id)
    if not _JOB_ID.fullmatch(value):
        raise ValueError("Invalid job ID")
    root = settings.jobs_dir.resolve()
    path = (root / value).resolve()
    if path.parent != root:
        raise ValueError("Invalid job path")
    return path
