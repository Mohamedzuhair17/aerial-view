import asyncio
import hmac
import logging
import shutil
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db
from app.api import analyze, chat, health, jobs
from app.config import settings
from app.cv.pipeline import Pipeline
from app.llm.client import LLM
from app.models.registry import load_models
from app.storage import job_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")
_rate_buckets: dict[str, deque] = defaultdict(deque)


def cleanup_expired_jobs():
    stale_cutoff = time.time() - settings.job_stale_hours * 3600
    marked = db.mark_stale_jobs(stale_cutoff)
    if marked:
        log.warning("Marked %d stale queued/running jobs as failed", marked)
    cutoff = time.time() - settings.job_retention_hours * 3600
    expired = db.delete_expired_jobs(cutoff)
    for job_id in expired:
        try:
            shutil.rmtree(job_directory(job_id), ignore_errors=True)
        except ValueError:
            log.error("Skipped filesystem cleanup for invalid stored job ID %r", job_id)
    if expired:
        log.info("Removed %d expired analysis jobs", len(expired))


async def _cleanup_loop():
    while True:
        await asyncio.sleep(max(60, settings.cleanup_interval_seconds))
        try:
            await asyncio.to_thread(cleanup_expired_jobs)
        except Exception:
            log.exception("Periodic job cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    cleanup_expired_jobs()
    app.state.queue_backend = settings.job_queue_backend
    if settings.job_queue_backend == "inline":
        app.state.pipe = Pipeline(load_models())
    else:
        # GPU/model memory belongs to the dedicated RQ worker, not the web process.
        app.state.pipe = None
    app.state.llm = LLM()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def security_middleware(request, call_next):
    # CORS preflight requests must reach CORSMiddleware without API-key challenges.
    if request.method == "OPTIONS":
        return await call_next(request)

    public_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if settings.api_key and request.url.path not in public_paths:
        supplied = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)

    if request.url.path not in public_paths:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _rate_buckets[client_ip]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= settings.rate_limit_per_minute:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        bucket.append(now)
        # Bound memory under high-cardinality source-IP traffic.
        if len(_rate_buckets) > settings.rate_limit_max_clients:
            stale = [key for key, values in _rate_buckets.items() if not values or now - values[-1] > 60]
            excess = max(0, len(_rate_buckets) - settings.rate_limit_max_clients)
            victims = stale[:excess]
            if len(victims) < excess:
                recent = sorted(
                    ((values[-1] if values else 0.0, key) for key, values in _rate_buckets.items()),
                    key=lambda item: item[0],
                )
                additional = [key for _, key in recent if key not in victims][:excess - len(victims)]
                victims.extend(additional)
            for key in victims:
                _rate_buckets.pop(key, None)

    return await call_next(request)


for route in (health.router, analyze.router, jobs.router, chat.router):
    app.include_router(route)
