# Aerial View Image Reasoning — Backend v2

FastAPI backend for paired pre-/post-disaster aerial imagery. It detects building instances, classifies building damage, stores image-space and (when available) geographic geometries, and answers questions using deterministic spatial tools with optional LLM-assisted tool selection.

> **Important:** This checkout now includes the five-class U-Net and Paddle ChangeFormer detector checkpoints. U-Net validation showed weak foreground damage recognition, so treat its masks as provisional. ChangeFormer boxes/classes are paired with U-Net masks; when no U-Net region overlaps a detection, the box is shown.

## Architecture

```text
Before + after images
        |
        +--> GeoTIFF CRS/affine reprojection (when both inputs are georeferenced)
        |       or conservative ORB + RANSAC image registration
        |
        +--> Paddle ChangeFormer detector (paired pre/post images)
        +--> U-Net damage masks refine detector boxes
        |
        +--> pixel polygons + damage classes + confidence + changed ratio
        |       + optional EPSG:4326 GeoJSON
        |
        +--> SQLite (local) or PostgreSQL + PostGIS (deployment)
        |
        +--> deterministic spatial tools <-- optional LLM planner
```

## Quick local run (SQLite, inline jobs)

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r requirements-geo.txt  # optional but recommended for GeoTIFF input
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`. Local defaults use SQLite and in-process background jobs. Configure `API_KEY` if the API is reachable by anyone other than the developer.

## Docker deployment (PostgreSQL + PostGIS + Redis + RQ)

1. Copy `.env.example` to `.env`.
2. Set a strong `POSTGRES_PASSWORD` and a separate random `API_KEY`. Because Compose builds `DATABASE_URL` from the same password variable, use URL-safe characters (for example, generate a 64-character hex password with `openssl rand -hex 32`).
3. Start the stack:

```bash
docker compose up --build -d
```

The stack contains:

- **PostGIS:** persistent relational and geographic data.
- **Redis:** durable queue broker.
- **API:** FastAPI web process; does not load GPU models in RQ mode.
- **Frontend:** Vite-built static site served by Nginx at `http://localhost:8080`; Nginx proxies `/api` to FastAPI and forwards `API_KEY` server-side.
- **Worker:** separate RQ process that loads the models and runs image analysis.

Open the frontend at `http://localhost:8080`. Check `http://localhost:8000/health` and `http://localhost:8000/docs` for API status and documentation. The health endpoint is intentionally public for container health checks; analysis data endpoints require `API_KEY` when it is configured. The frontend proxy adds this key server-side.

To view logs:

```bash
docker compose logs -f frontend api worker db redis
```

To stop without deleting data:

```bash
docker compose down
```

To delete persistent database, Redis and image volumes (destructive):

```bash
docker compose down -v
```

## Database configuration

### SQLite development

```env
DATABASE_BACKEND=sqlite
DATABASE_URL=
JOB_QUEUE_BACKEND=inline
```

### PostgreSQL + PostGIS

```env
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/aerial_db
```

The API initializes PostGIS and the required tables/indexes on startup. The schema includes:

- `jobs`: job state, image dimensions, model mode, creation time and alignment metadata.
- `buildings`: per-building damage data, pixel-space JSON geometry, optional PostGIS `geom`.
- GiST index on `buildings.geom` and B-tree indexes for job lookup/retention.

The `geom` column uses SRID 4326. It is populated only when the backend has explicit georeferencing metadata and can convert pixel polygons to longitude/latitude. Ordinary PNG/JPG uploads keep their polygons in pixel coordinates and leave `geom` NULL. Pixel coordinates are never relabelled as GPS coordinates.

### Migrate an existing SQLite database

Stop writes to the source API and back up both databases first. Start PostGIS, then run from the project root with PostgreSQL settings in `.env` or the shell:

```bash
DATABASE_BACKEND=postgres \
DATABASE_URL='postgresql://USER:PASSWORD@localhost:5432/aerial_db' \
python scripts/migrate_sqlite_to_postgres.py --sqlite data/app.db
```

The migration is repeatable for existing job IDs. It refreshes each migrated job's metadata and building rows, including clearing the target building set when the source job has no detections. By default it also copies each `data/jobs/<job-id>/` folder (before/after/overlay images and metadata) into the target `DATA_DIR/jobs/`; use `--skip-files` for database-only migration or `--jobs-dir` if the source artifacts live elsewhere. It does not delete unrelated target jobs. Pixel-space polygons remain JSON; only validated EPSG:4326 GeoJSON populates PostGIS geometry. Check the warning count for any missing source job folders before switching traffic.

## GeoTIFF and geospatial processing

Install `requirements-geo.txt` for Rasterio support. Upload `.tif` or `.tiff` files with embedded CRS and affine metadata.

- The after-image raster defines the output grid and geographic footprint.
- If both images have CRS/affine metadata, the before image is reprojected onto the after image grid.
- If only one image is georeferenced, the service falls back to image feature registration; only the after image's georeferencing can be used to map output building polygons.
- GeoTIFF pixel scaling for model input is a per-band 2nd–98th percentile stretch. It is not radiometric calibration.
- Building contours are transformed through the raster affine transform and reprojected to EPSG:4326 before PostGIS insertion.

The PostGIS query helper accepts GeoJSON Geometry coordinates in EPSG:4326:

```python
from app.db import find_buildings_intersecting

matches = find_buildings_intersecting(
    {"type": "Polygon", "coordinates": [[[80.0, 13.0], [80.1, 13.0],
      [80.1, 13.1], [80.0, 13.1], [80.0, 13.0]]]},
    job_id="optional-job-id",
)
```

## Image registration and quality

The service reports alignment metadata on `POST /analyze` and `GET /jobs/{id}`.

- GeoTIFF pairs with CRS/affine metadata use raster reprojection.
- Other pairs use ORB feature matching and RANSAC homography with checks for feature count, inlier ratio, plausible projected area and image overlap.
- Weak alignment is reported as `unverified`; it is not silently labelled aligned.
- Set `REQUIRE_ALIGNMENT=true` to reject pairs when alignment cannot be verified. This is recommended for high-stakes use.

Feature registration is not a substitute for orthorectification, ground control points, or careful acquisition consistency. Large parallax, different sensors, seasonal change and major destruction can still make automatic alignment unreliable.

## Model integration

`MODEL_MODE` supports:

- `stub`: deterministic simulated models for API/frontend development.
- `auto`: use a real model when its configured checkpoint exists; otherwise use a stub per model.
- `real`: require real checkpoints; startup fails if dependencies or checkpoints are missing.

Expected checkpoint files:

```text
weights/changeformer.pt            TorchScript ChangeFormer
weights/maskrcnn_buildings.pth     torchvision Mask R-CNN state_dict (2 classes)
weights/effnet_b0_damage.pth       {"model": state_dict, "classes": [...], "six_channel": true}
runs/unet_train_only/checkpoints/unet_best.pth  five-class U-Net checkpoint (auto-selected when present)
runs/unet_train_only/checkpoints/best_changeformer_detector.pdparams  Paddle ChangeFormer detector
```

Set `UNET_WEIGHTS`, `UNET_INPUT_SIZE`, `CHANGE_WEIGHTS`, and `CHANGE_INPUT_SIZE` to override the checkpoint defaults. The frontend's **Model stack** field reports the loaded inference path.

The supplied U-Net predicts a semantic damage map; connected same-class pixels are emitted as regions, so touching buildings may merge. The ChangeFormer detector predicts damage-labeled boxes; overlapping U-Net regions refine their shapes. The displayed pixel-difference ratio is a simple image-difference heuristic, not a ChangeFormer probability map. This is not the TorchScript pixel-change-mask interface described below.

Install `requirements-ml.txt` for PyTorch/TorchVision and `requirements-paddle.txt` for ChangeFormer. On Windows with CUDA 11.8, install Paddle GPU with `python -m pip install -r requirements-paddle.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/`. Set `DEVICE=cuda`; PyTorch and Paddle must both see the GPU. The Paddle checkpoint adapter raises a clear startup error if the requested CUDA device is unavailable.

### TorchScript ChangeFormer tile contract (legacy `.pt` checkpoints)

The serving wrapper assumes the TorchScript model accepts two RGB tensors of shape `[1, 3, 256, 256]`, normalized to `[-1, 1]`, and returns two-class logits `[1, 2, H, W]` (or a list/tuple whose last item is logits). `CHANGE_TILE_SIZE` must remain 256 for this checkpoint contract. Inference uses overlapping source-image tiles and blends class probabilities; it does not resize a 512px source tile down to 256px. If your trained model has a different input size or output structure, update the wrapper and settings together.

The `.pdparams` ChangeFormer detector instead uses the Paddle architecture in `app/models/changeformer_paddle.py`. It proposes damage-labeled boxes from paired images and returns detector classes plus scores. `CHANGE_INPUT_SIZE` must match the size used in training (the checked-in `.env` uses 512); set `DEVICE=cuda` and install the Windows CUDA 11.8 Paddle wheel to run this model on the GPU.

### Damage classifier contract

EfficientNet-B0 supports either six-channel before+after crops or three-channel after-only crops. Checkpoint classes must be exactly the four supported classes: `No Damage`, `Minor`, `Major`, `Destroyed`. `Unclassified` is assigned by the confidence threshold or invalid model output. Confidence is not calibrated unless a temperature has been fitted on a held-out labelled validation set.

```bash
python scripts/calibrate_classifier.py validation_logits.npz --output temperature.json
```

Copy the fitted `temperature` value into the classifier checkpoint metadata. Calibration does not improve classification accuracy; it only adjusts probability confidence.

## Job queue and retention

- Local default: `JOB_QUEUE_BACKEND=inline` uses FastAPI background tasks. Suitable for demos and single-process development only.
- Deployment: `JOB_QUEUE_BACKEND=rq` sends jobs to Redis; the separate worker owns model memory and executes the analysis. Docker Compose configures this mode.
- Queued/running jobs older than `JOB_STALE_HOURS` are marked failed; jobs older than `JOB_RETENTION_HOURS` are periodically removed from the database and shared job-image volume.
- For multiple API replicas, use a shared Redis instance, shared object storage for image artifacts, and a distributed rate limiter/API gateway. The included rate limiter is process-local.

## Security and limits

- Uploads are read in bounded chunks and constrained by `MAX_UPLOAD_MB`, `MAX_IMAGE_SIDE` and `MAX_IMAGE_PIXELS`.
- Optional API-key authentication supports `X-API-Key` and Bearer headers. `/health` and API documentation remain public; protect docs at the network layer if required.
- CORS is restricted to `CORS_ORIGINS`; OPTIONS preflight is allowed through authentication middleware.
- Per-process rate limiting is a basic safeguard, not a distributed abuse-prevention system.
- Do not expose Redis or PostgreSQL ports publicly in production. Use secrets management, TLS, network policies and backups.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | API, database mode, queue mode and model execution status |
| POST | `/analyze` | Multipart `before` and `after` uploads; returns a job ID |
| GET | `/jobs/{id}` | Job status and alignment metadata |
| GET | `/jobs/{id}/buildings` | Building-level damage records; optional damage/quadrant filters |
| GET | `/jobs/{id}/summary` | Damage totals and quadrant matrix |
| GET | `/jobs/{id}/image/{before\|after\|overlay}` | Stored source or overlay image |
| GET | `/jobs/{id}/overlay` | Rendered overlay with optional highlighted building IDs |
| POST | `/chat` | Deterministic spatial answer with optional LLM tool planning |

Example:

```bash
JOB_ID=$(curl -s -F before=@before.png -F after=@after.png http://localhost:8000/analyze | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
curl -s http://localhost:8000/jobs/$JOB_ID
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d "{\"job_id\":\"$JOB_ID\",\"question\":\"Which quadrant has the most destroyed buildings?\"}"
```

## Tests

```bash
python -m pytest -q
```

The normal test suite exercises SQLite, stub inference, API security/CORS, alignment checks, tile-size behavior, geometry validation and geospatial polygon conversion when Rasterio is installed. A real PostGIS integration test is skipped unless `TEST_POSTGRES_URL` points to a disposable PostGIS database.

## Known limitations

- No trained checkpoints or model accuracy evidence are included in this source archive.
- ORB/RANSAC alignment is a fallback, not guaranteed registration for high-resolution satellite imagery.
- GeoTIFF percentile stretching is a display/model-input transform; it does not preserve calibrated spectral reflectance.
- The backward-compatible pixel overlay contours preserve external contours but not interior holes; GeoTIFF-derived PostGIS geometries are polygonized directly from masks and preserve interior rings.
- The pipeline still processes the full image with Mask R-CNN; very large scenes may need tiled instance segmentation and cross-tile instance merging.
- RQ improves job durability, but this project does not include multi-tenant authorization, object-storage integration, distributed rate limiting, Alembic migrations or production observability.
- All AI outputs require human review before operational disaster-response use.
