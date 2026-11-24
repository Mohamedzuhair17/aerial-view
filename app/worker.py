"""Dedicated RQ worker. Start with: python -m app.worker"""
from functools import lru_cache

from redis import Redis
from rq import Queue, Worker

from app import db
from app.config import settings
from app.cv.pipeline import Pipeline
from app.models.registry import load_models


@lru_cache(maxsize=1)
def _pipeline():
    return Pipeline(load_models())


def process_job(job_id: str):
    from app.api.analyze import _run_job
    current = db.get_job(job_id)
    if current is None or current["status"] in {"done", "failed"}:
        return
    try:
        pipe = _pipeline()
    except Exception as exc:
        # A model/dependency failure must not leave the API job stuck in queued state.
        try:
            db.update_job(job_id, status="failed", error=f"Worker initialization failed: {exc}"[:500])
        except Exception:
            pass
        raise
    _run_job(pipe, job_id)


def enqueue_job(job_id: str):
    connection = Redis.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3)
    connection.ping()
    queue = Queue(settings.redis_queue_name, connection=connection)
    queue.enqueue(process_job, job_id, job_timeout="2h", result_ttl=3600, failure_ttl=86400)


def main():
    db.init_db()
    connection = Redis.from_url(settings.redis_url)
    connection.ping()
    queue = Queue(settings.redis_queue_name, connection=connection)
    Worker([queue], connection=connection).work(with_scheduler=False)


if __name__ == "__main__":
    main()
