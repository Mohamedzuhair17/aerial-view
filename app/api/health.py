from fastapi import APIRouter, HTTPException, Request

from app import db
from app.config import settings

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(request: Request):
    try:
        database = db.ping()
    except Exception as exc:
        raise HTTPException(503, f"Database is unavailable: {type(exc).__name__}") from exc

    queue_status = {"backend": settings.job_queue_backend}
    if settings.job_queue_backend == "rq":
        try:
            from redis import Redis
            from rq import Queue
            connection = Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
            connection.ping()
            queue_status["connected"] = True
            queue_status["queued_jobs"] = Queue(settings.redis_queue_name, connection=connection).count
            from rq import Worker
            queue_status["active_workers"] = len(Worker.all(connection=connection))
        except Exception as exc:
            raise HTTPException(503, f"Job queue is unavailable: {type(exc).__name__}") from exc

    pipe = getattr(request.app.state, "pipe", None)
    llm = request.app.state.llm
    return {
        "status": "ok",
        "execution": settings.job_queue_backend,
        "models": pipe.models.status if pipe is not None else {"worker": "dedicated RQ worker"},
        "model_mode": pipe.mode if pipe is not None else settings.model_mode,
        "llm": {"provider": llm.provider, "model": llm.model, "active": llm.available},
        "database": database,
        "queue": queue_status,
    }
