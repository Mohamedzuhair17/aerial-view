import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.llm.agent import rule_plan


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings.data_dir = tmp_path_factory.mktemp("data")
    settings.model_mode = "stub"
    settings.llm_provider = "none"
    from app.main import app

    with TestClient(app) as c:
        yield c


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(scope="module")
def job_id(client):
    rng = np.random.default_rng(1)
    before = rng.integers(60, 140, (480, 640, 3), dtype=np.uint8)
    after = before.copy()
    for _ in range(40):  # simulate damage patches
        x, y = int(rng.integers(0, 580)), int(rng.integers(0, 420))
        after[y:y + 60, x:x + 60] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    r = client.post("/analyze", files={"before": ("b.png", _png(before), "image/png"),
                                       "after": ("a.png", _png(after), "image/png")})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    for _ in range(50):
        s = client.get(f"/jobs/{jid}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert s["status"] == "done", s
    return jid


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_mode"] == "stub"


def test_buildings_and_summary(client, job_id):
    rows = client.get(f"/jobs/{job_id}/buildings").json()
    assert len(rows) > 0
    assert {r["quadrant"] for r in rows} <= {"NW", "NE", "SW", "SE"}
    s = client.get(f"/jobs/{job_id}/summary").json()
    assert s["total"] == len(rows) == sum(s["by_damage"].values())
    assert sum(sum(v.values()) for v in s["by_quadrant"].values()) == len(rows)
    assert s["alignment"] is not None


def test_quadrant_question_matches_data(client, job_id):
    rows = client.get(f"/jobs/{job_id}/buildings").json()
    r = client.post("/chat", json={"job_id": job_id, "question": "Which quadrant has the most destroyed buildings?"}).json()
    assert r["tool"] == "quadrant_ranking" and r["reasoning_mode"] == "rules"
    manual = {q: sum(1 for b in rows if b["quadrant"] == q and b["damage"] == "Destroyed") for q in ("NW", "NE", "SW", "SE")}
    assert r["evidence"]["counts"] == manual
    assert set(r["highlight_ids"]) <= {b["id"] for b in rows}


def test_count_question(client, job_id):
    rows = client.get(f"/jobs/{job_id}/buildings").json()
    r = client.post("/chat", json={"job_id": job_id, "question": "How many buildings are destroyed in the left half?"}).json()
    exp = sum(1 for b in rows if b["damage"] == "Destroyed" and b["quadrant"].endswith("W"))
    assert r["tool"] == "count_buildings" and r["evidence"]["count"] == exp


def test_unanswerable_and_errors(client, job_id):
    r = client.post("/chat", json={"job_id": job_id, "question": "How many people died?"}).json()
    assert r["tool"] == "unanswerable" and r["highlight_ids"] == []
    assert client.post("/chat", json={"job_id": "nope", "question": "hello?"}).status_code == 404
    assert client.post("/analyze", files={"before": ("b.png", b"junk"), "after": ("a.png", b"junk")}).status_code == 400


def test_overlay_png(client, job_id):
    r = client.get(f"/jobs/{job_id}/overlay", params={"highlight": "B001,B002"})
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(f"/jobs/{job_id}/image/overlay").status_code == 200


@pytest.mark.parametrize("q,tool", [
    ("compare left vs right", "compare_regions"),
    ("breakdown of damage in the northeast", "damage_distribution"),
    ("show me the major buildings in the bottom half", "list_buildings"),
    ("tell me about B12", "building_details"),
    ("how many buildings are severely damaged", "count_buildings"),
])
def test_rule_router(q, tool):
    assert rule_plan(q)[0] == tool


def test_polygon_components_are_returned(client, job_id):
    rows = client.get(f"/jobs/{job_id}/buildings").json()
    assert rows
    for row in rows:
        assert "polygons" in row and isinstance(row["polygons"], list)
        assert row["polygons"]
        assert row["polygon"] == row["polygons"][0]


def test_upload_response_reports_alignment(client):
    rng = np.random.default_rng(22)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    response = client.post("/analyze", files={"before": ("b.png", _png(img), "image/png"),
                                              "after": ("a.png", _png(img), "image/png")})
    assert response.status_code == 202
    assert response.json()["alignment"]["status"] in {"aligned", "unverified", "disabled"}


def test_sqlite_db_roundtrip_and_pixel_geometry_is_not_geographic(client):
    from app import db
    jid = "db-roundtrip-test"
    db.create_job(jid, 100, 80)
    db.save_buildings(jid, [{
        "id": "B001", "bbox": [1, 2, 30, 40], "polygon": [[1, 2], [30, 2], [30, 40]],
        "polygons": [[[1, 2], [30, 2], [30, 40]]], "damage": "Minor", "conf": 0.8,
        "changed_ratio": 0.2, "quadrant": "NW", "area": 100,
    }])
    result = db.get_buildings(jid)
    assert result[0]["id"] == "B001"
    assert result[0]["polygon"][0] == [1, 2]
    assert result[0]["geometry_geojson"] is None


def test_postgis_spatial_query_is_explicitly_unavailable_in_sqlite():
    from app import db
    with pytest.raises(RuntimeError, match="require DATABASE_BACKEND=postgres"):
        db.find_buildings_intersecting({"type": "Point", "coordinates": [0, 0]})


def test_job_alignment_metadata_is_persisted(client, job_id):
    status = client.get(f"/jobs/{job_id}").json()
    assert "alignment" in status
    assert status["alignment"]["status"] in {"aligned", "unverified", "disabled"}


def test_cors_preflight_is_not_blocked_by_api_key(client):
    old_key = settings.api_key
    settings.api_key = "test-secret"
    try:
        response = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-api-key",
            },
        )
        assert response.status_code in (200, 204)
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    finally:
        settings.api_key = old_key


def test_api_key_protects_routes_but_health_remains_public(client):
    old_key = settings.api_key
    settings.api_key = "test-secret"
    try:
        assert client.get("/health").status_code == 200
        assert client.get("/jobs/not-a-job").status_code == 401
        assert client.get("/jobs/not-a-job", headers={"X-API-Key": "test-secret"}).status_code == 404
    finally:
        settings.api_key = old_key


def test_geojson_validation_rejects_pixel_coordinates_and_nan():
    from app.db import _validate_geojson_geometry
    import pytest
    with pytest.raises(ValueError, match="EPSG:4326"):
        _validate_geojson_geometry({"type": "Point", "coordinates": [500, 700]})
    with pytest.raises(ValueError):
        _validate_geojson_geometry({"type": "Point", "coordinates": [float("nan"), 10]})
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_geojson_geometry({"type": "Polygon", "coordinates": []})


def test_expired_job_cleanup_removes_building_rows(client):
    from app import db
    jid = "expired-job-test"
    db.create_job(jid, 20, 20)
    db.save_buildings(jid, [{
        "id": "B001", "bbox": [1, 1, 10, 10], "polygon": [[1, 1], [10, 1], [10, 10]],
        "polygons": [[[1, 1], [10, 1], [10, 10]]], "damage": "Minor", "conf": 0.8,
        "changed_ratio": 0.2, "quadrant": "NW", "area": 30,
    }])
    with db.conn() as connection:
        connection.execute("UPDATE jobs SET created_at=? WHERE id=?", (0.0, jid))
    deleted = db.delete_expired_jobs(1.0)
    assert jid in deleted
    assert db.get_job(jid) is None
    assert db.get_buildings(jid) == []


def test_list_buildings_limit_is_bounded():
    from app.spatial.tools import list_buildings
    import pytest
    with pytest.raises(ValueError, match="between 1 and 100"):
        list_buildings([], limit=1000)


def test_registration_rejects_aspect_ratio_mismatch():
    from app.api.analyze import _register_before
    before = np.zeros((100, 200, 3), dtype=np.uint8)
    after = np.zeros((100, 100, 3), dtype=np.uint8)
    _, metadata = _register_before(before, after)
    assert metadata["status"] == "unverified"
    assert metadata["reason"] == "aspect_ratio_mismatch"


def test_geotiff_pixel_polygons_convert_to_wgs84_when_rasterio_available():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from app.cv.geospatial import attach_geographic_geometries
    rows = [{
        "id": "B001", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "polygons": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
    }]
    # UTM zone 44N around Chennai; pixel coordinates must become lon/lat GeoJSON.
    transform = from_origin(500000, 1400000, 10, 10)
    georef = {"crs": "EPSG:32644", "affine": list(transform)[:6], "georeferenced": True}
    attach_geographic_geometries(rows, georef)
    geom = rows[0]["geometry_geojson"]
    assert geom["type"] == "MultiPolygon"
    lon, lat = geom["coordinates"][0][0][0]
    assert 70 < lon < 90 and 0 < lat < 30


def test_sqlite_expired_job_cleanup_and_alignment_roundtrip(client):
    from app import db
    jid = "alignment-roundtrip"
    alignment = {"status": "aligned", "matches": 100, "inliers": 80}
    db.create_job(jid, 100, 80, alignment=alignment)
    assert db.get_job(jid)["alignment"] == alignment


def test_changeformer_tiles_at_checkpoint_resolution(monkeypatch):
    import torch
    from app.models.real import RealChange

    class ConstantModel:
        def __init__(self):
            self.seen = []
        def __call__(self, before, after):
            self.seen.append((tuple(before.shape), tuple(after.shape)))
            logits = torch.zeros((1, 2, 256, 256), dtype=torch.float32)
            logits[:, 1] = 1.0
            return logits

    monkeypatch.setattr(settings, "change_tile_size", 256)
    monkeypatch.setattr(settings, "change_tile_overlap", 32)
    model = RealChange.__new__(RealChange)
    model.model = ConstantModel()
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    result = model.predict(image, image)
    assert result.shape == (300, 300)
    assert result.all()
    assert model.model.seen
    assert all(shape == (1, 3, 256, 256) for pair in model.model.seen for shape in pair)


def test_changeformer_rejects_tiles_larger_than_checkpoint_input(monkeypatch):
    from app.models.real import RealChange
    monkeypatch.setattr(settings, "change_tile_size", 512)
    model = RealChange.__new__(RealChange)
    model.model = lambda before, after: None
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="must equal the checkpoint input size"):
        model.predict(image, image)


def test_rate_limiter_bounds_high_cardinality_client_map(client, monkeypatch):
    import app.main as main_module
    old_limit = settings.rate_limit_max_clients
    settings.rate_limit_max_clients = 2
    main_module._rate_buckets.clear()
    try:
        for ip in ("198.51.100.1", "198.51.100.2", "198.51.100.3", "198.51.100.4"):
            main_module._rate_buckets[ip].append(time.monotonic())
        # A request should trigger eviction without raising and keep the map bounded.
        response = client.get("/jobs/not-a-job")
        assert response.status_code == 404
        assert len(main_module._rate_buckets) <= 2
    finally:
        settings.rate_limit_max_clients = old_limit
        main_module._rate_buckets.clear()


def test_analyze_rejects_incompatible_image_footprints(client):
    before = np.zeros((100, 200, 3), dtype=np.uint8)
    after = np.zeros((100, 100, 3), dtype=np.uint8)
    response = client.post(
        "/analyze",
        files={"before": ("before.png", _png(before), "image/png"),
               "after": ("after.png", _png(after), "image/png")},
    )
    assert response.status_code == 422
    assert "aspect ratios" in response.json()["detail"]


def test_sql_placeholder_translation_for_postgres(monkeypatch):
    from app import db
    monkeypatch.setattr(settings, "database_backend", "postgres")
    assert db.sql("SELECT * FROM jobs WHERE id=? AND status=?") == "SELECT * FROM jobs WHERE id=%s AND status=%s"


def test_existing_sqlite_schema_is_upgraded_without_data_loss(tmp_path, monkeypatch):
    import sqlite3
    from app import db
    monkeypatch.setattr(settings, "database_backend", "sqlite")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    old = sqlite3.connect(tmp_path / "app.db")
    old.executescript("""
        CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT NOT NULL,error TEXT,width INTEGER,height INTEGER,n_buildings INTEGER,model_mode TEXT,created_at REAL);
        CREATE TABLE buildings(job_id TEXT NOT NULL,id TEXT NOT NULL,bbox TEXT,polygon TEXT,damage TEXT,conf REAL,changed_ratio REAL,quadrant TEXT,area INTEGER,PRIMARY KEY(job_id,id));
        INSERT INTO jobs(id,status,width,height,created_at) VALUES('legacy','done',20,20,1.0);
        INSERT INTO buildings(job_id,id,bbox,polygon,damage,conf,changed_ratio,quadrant,area) VALUES('legacy','B001','[1,2,3,4]','[[1,2],[3,2],[3,4]]','Minor',0.8,0.2,'NW',10);
    """)
    old.commit()
    old.close()
    db.init_db()
    assert db.get_job("legacy")["status"] == "done"
    result = db.get_buildings("legacy")
    assert result[0]["id"] == "B001"
    assert result[0]["polygons"] == [result[0]["polygon"]]
    assert result[0]["geometry_geojson"] is None


def test_stale_jobs_are_marked_failed(client):
    from app import db
    jid = "stale-job-test"
    db.create_job(jid, 10, 10)
    with db.conn() as connection:
        connection.execute("UPDATE jobs SET created_at=? WHERE id=?", (0.0, jid))
    changed = db.mark_stale_jobs(1.0)
    assert changed >= 1
    assert db.get_job(jid)["status"] == "failed"


def test_geotiff_decode_and_reprojection_when_rasterio_available():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from app.api.analyze import _decode
    from app.cv.geospatial import reproject_before_to_after

    def make_tiff(value, x_origin):
        transform = from_origin(x_origin, 1400000, 10, 10)
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=64, height=64, count=3, dtype="uint16",
                          crs="EPSG:32644", transform=transform) as dataset:
                dataset.write(np.full((3, 64, 64), value, dtype=np.uint16))
            return mem.read()

    before, before_meta = _decode(make_tiff(1000, 500000), "before", "before.tif")
    after, after_meta = _decode(make_tiff(5000, 500000), "after", "after.tif")
    assert before_meta["georeferenced"] and after_meta["georeferenced"]
    assert before.shape == after.shape == (64, 64, 3)
    aligned, overlap = reproject_before_to_after(before, before_meta, after_meta, after.shape[:2])
    assert aligned.shape == after.shape and overlap > 0.99
    assert int(before.mean()) < int(after.mean())


def test_geotiff_polygonization_uses_mask_and_preserves_holes_when_available():
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from app.cv.geospatial import attach_geographic_geometries
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True
    mask[7:13, 7:13] = False
    row = {
        "id": "B001", "bbox": [2, 2, 18, 18],
        "polygon": [[2, 2], [17, 2], [17, 17], [2, 17]],
        "polygons": [[[2, 2], [17, 2], [17, 17], [2, 17]]],
        "_mask": mask,
    }
    transform = from_origin(500000, 1400000, 10, 10)
    georef = {"crs": "EPSG:32644", "affine": list(transform)[:6], "georeferenced": True}
    attach_geographic_geometries([row], georef)
    assert "_mask" not in row
    geometry = row["geometry_geojson"]
    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"]) == 2  # exterior ring + interior hole


def test_job_directory_rejects_path_traversal():
    from app.storage import job_directory
    import pytest
    with pytest.raises(ValueError, match="Invalid job ID"):
        job_directory("../../etc")
