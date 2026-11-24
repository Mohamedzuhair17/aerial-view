"""Optional real PostGIS integration tests.

Set TEST_POSTGRES_URL to a disposable PostGIS database to enable this test.
"""
import os
import uuid

import pytest

POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL not configured")


def test_postgis_geometry_roundtrip_and_intersection(monkeypatch, tmp_path):
    from app import db
    from app.config import settings

    monkeypatch.setattr(settings, "database_backend", "postgres")
    monkeypatch.setattr(settings, "database_url", POSTGRES_URL)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    db.init_db()

    job_id = "postgis-test-" + uuid.uuid4().hex[:12]
    db.create_job(job_id, 10, 10)
    geometry = {
        "type": "Polygon",
        "coordinates": [[[80.0, 13.0], [80.01, 13.0], [80.01, 13.01], [80.0, 13.01], [80.0, 13.0]]],
    }
    db.save_buildings(job_id, [{
        "id": "B001", "bbox": [0, 0, 10, 10], "polygon": [[0, 0], [10, 0], [10, 10]],
        "polygons": [[[0, 0], [10, 0], [10, 10]]], "damage": "Major", "conf": 0.9,
        "changed_ratio": 0.8, "quadrant": "NE", "area": 50, "geometry_geojson": geometry,
    }])
    roundtrip = db.get_buildings(job_id)
    assert roundtrip[0]["geometry_geojson"]["type"] in {"Polygon", "MultiPolygon"}
    found = db.find_buildings_intersecting({
        "type": "Point", "coordinates": [80.005, 13.005]
    }, job_id=job_id)
    assert [row["id"] for row in found] == ["B001"]
    with db.conn() as connection:
        connection.execute("DELETE FROM buildings WHERE job_id=%s", (job_id,))
        connection.execute("DELETE FROM jobs WHERE id=%s", (job_id,))
