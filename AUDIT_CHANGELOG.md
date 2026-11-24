# Aerial Backend v2 — Deep Audit and Upgrade Log

## Critical defects corrected

1. **PostgreSQL startup cleanup used SQLite `?` placeholders.** Centralized SQL placeholder translation and moved cleanup into backend-aware database helpers.
2. **ChangeFormer 512px tiles were resized to a 256px checkpoint input.** The serving contract is now explicit: source tiles must be 256px for the supplied 256px TorchScript interface. Partial edge tiles are padded, overlapping class probabilities are blended, and larger configured tiles fail loudly instead of silently downsampling.
3. **Image registration could accept weak transforms.** Added minimum feature/inlier checks, projected-area sanity checks, overlap checks, exception handling, and persisted homography/alignment metadata. Incompatible aspect ratios are rejected before the job is queued. `REQUIRE_ALIGNMENT=true` can reject unverified pairs.
4. **CORS preflight could be blocked by API-key middleware.** OPTIONS requests now pass through to CORS middleware. API-key comparison uses constant-time comparison; rate-limit client storage is bounded.
5. **SQLite-to-PostGIS migration did not reliably refresh empty building sets.** Migration now upserts job metadata, replaces all building rows including empty sets, copies alignment metadata where present, and can copy associated job image/artifact folders.
6. **GeoJSON could be incorrectly or unsafely stored.** Added geometry type, coordinate-range, finite-number and PostGIS validity checks. Pixel coordinates remain JSON and are never assigned SRID 4326.
7. **GeoTIFF handling was absent.** Added optional Rasterio ingestion, CRS/affine extraction, reprojection of before-image onto the after-image grid, overlap validation, and pixel-polygon conversion to EPSG:4326. GeoTIFF model scaling uses a configurable fixed range shared between acquisitions.
8. **Upload processing had incomplete resource checks.** Added bounded chunk reads, image-header dimension checks, reduced-resolution decode where supported, maximum pixel checks and explicit write/decode failures.
9. **Worker/job failures could leave jobs queued or running indefinitely.** Added optional Redis/RQ worker separation, worker initialization failure handling, stale-job failure marking and periodic retention cleanup.
10. **Model confidence/checkpoint handling was weak.** Added safe tensor-only checkpoint loading where supported, class-order validation, CUDA availability checks, finite temperature checks and invalid-confidence handling.
11. **LLM output could influence the final numeric answer.** LLM is restricted to tool selection; spatial tools generate final deterministic answer text. Explicitly unsupported question categories are rejected before LLM routing.

## Other improvements

- Segmentation-mask crops use a building-pixel median fill outside the mask rather than artificial black pixels.
- All external contours are preserved through a typed `BuildingInstance` field; primary polygon remains backward compatible.
- Overlay image encoding failures return a controlled API error.
- Generated job IDs are validated before filesystem access to prevent path traversal through corrupted/imported database IDs.
- Invalid quadrant filters and unbounded LLM list limits are rejected.
- Health endpoint checks database/PostGIS and Redis queue connectivity.
- Docker Compose now includes PostGIS, Redis, API and a dedicated RQ worker; database/API host ports bind to localhost by default.
- Added `.dockerignore`, optional ML build flag, GeoTIFF requirements and safer model-weight download validation.
- Calibration script validates logits/labels and reports NLL/ECE before and after temperature scaling.

## Validation performed in this environment

- Python compilation: passed.
- SQLite and stub-mode API tests: **34 passed**.
- Optional PostGIS integration test: **1 skipped** because `TEST_POSTGRES_URL` was not configured.
- GeoTIFF decode/reprojection/polygon conversion tests: passed when Rasterio was available.
- Calibration script smoke test: passed.
- Migration script CLI help: passed.
- Docker Compose YAML syntax parsed successfully.

## Not verified / remaining work

- Docker itself is not installed in the audit environment, so an image build and `docker compose up` were not executed.
- No PostgreSQL/PostGIS server or psycopg driver was available in the audit environment; the real database integration test remains to be run against a disposable PostGIS instance.
- Redis/RQ runtime integration was not executed against a live Redis service.
- The archive still contains no trained ChangeFormer, Mask R-CNN or EfficientNet weights. Stub mode is not real AI inference and has no disaster-damage accuracy claim.
- Model checkpoint compatibility must be verified with the exact training code, preprocessing and class mapping used to produce the weights.
- ORB/RANSAC is a fallback and is not a substitute for orthorectification or ground-control-point registration. For high-stakes use, set `REQUIRE_ALIGNMENT=true` and require human review.
- GeoTIFF stretch values must be configured for the sensor/product; this is not radiometric calibration.
- The backward-compatible pixel contour representation does not retain interior holes; GeoTIFF/PostGIS geometries are generated from segmentation masks and preserve interior rings.
- Production deployment still needs external secrets management, TLS, backups, distributed rate limiting, multi-tenant authorization, observability, and formal versioned schema migrations (e.g. Alembic).
