"""SQLite/PostgreSQL persistence with safe PostGIS geometry handling.

Pixel-space geometry is stored as JSON. The PostGIS geom column is populated only
from explicit GeoJSON geometry in EPSG:4326 (longitude/latitude).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator

from app.config import settings

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT,
  width INTEGER, height INTEGER, n_buildings INTEGER, model_mode TEXT,
  created_at REAL NOT NULL, alignment_json TEXT
);
CREATE TABLE IF NOT EXISTS buildings(
  job_id TEXT NOT NULL, id TEXT NOT NULL, bbox TEXT, polygon TEXT, damage TEXT,
  conf REAL, changed_ratio REAL, quadrant TEXT, area INTEGER, polygons TEXT,
  geometry_geojson TEXT, PRIMARY KEY(job_id, id)
);
CREATE INDEX IF NOT EXISTS idx_b_job ON buildings(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
"""

POSTGRES_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, width INTEGER, height INTEGER, n_buildings INTEGER, model_mode TEXT, created_at DOUBLE PRECISION NOT NULL, alignment_json TEXT)",
    "CREATE TABLE IF NOT EXISTS buildings(job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, id TEXT NOT NULL, bbox TEXT, polygon TEXT, damage TEXT, conf DOUBLE PRECISION, changed_ratio DOUBLE PRECISION, quadrant TEXT, area BIGINT, polygons TEXT, geom geometry(Geometry,4326), PRIMARY KEY(job_id,id))",
    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS alignment_json TEXT",
    "ALTER TABLE buildings ADD COLUMN IF NOT EXISTS geom geometry(Geometry,4326)",
    "CREATE INDEX IF NOT EXISTS idx_b_job ON buildings(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_buildings_geom ON buildings USING GIST(geom)",
)


def is_postgres() -> bool:
    return settings.database_backend.lower() in {"postgres", "postgresql", "postgis"}


def sql(statement: str) -> str:
    """Translate DB-API qmark placeholders for psycopg; SQL text is authored internally."""
    return statement.replace("?", "%s") if is_postgres() else statement


@contextmanager
def conn() -> Iterator:
    if not is_postgres():
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(settings.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        try:
            c.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
    else:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is required when DATABASE_BACKEND=postgres")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg; install requirements.txt") from exc
        c = psycopg.connect(settings.database_url, row_factory=dict_row, connect_timeout=10)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _json_text(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False) if value is not None else None


def _validate_geojson_geometry(geometry: dict) -> str:
    if not isinstance(geometry, dict) or geometry.get("type") not in {
        "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"
    }:
        raise ValueError("geometry_geojson must be a GeoJSON Geometry object")
    if "crs" in geometry:
        raise ValueError("GeoJSON CRS member is not supported; coordinates must be EPSG:4326")

    def check_coords(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            lon, lat = float(value[0]), float(value[1])
            if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("GeoJSON coordinates must be finite EPSG:4326 longitude/latitude")
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                check_coords(child)
        else:
            raise ValueError("Invalid GeoJSON coordinates")

    if geometry["type"] == "GeometryCollection":
        geoms = geometry.get("geometries")
        if not isinstance(geoms, list) or not geoms:
            raise ValueError("GeometryCollection must contain geometries")
        for child in geoms:
            _validate_geojson_geometry(child)
    else:
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or not coordinates:
            raise ValueError("GeoJSON coordinates cannot be empty")
        check_coords(coordinates)
    return _json_text(geometry)


def init_db():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        if not is_postgres():
            c.executescript(SQLITE_SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(buildings)").fetchall()}
            if "polygons" not in cols:
                c.execute("ALTER TABLE buildings ADD COLUMN polygons TEXT")
            if "geometry_geojson" not in cols:
                c.execute("ALTER TABLE buildings ADD COLUMN geometry_geojson TEXT")
            jcols = {r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
            if "alignment_json" not in jcols:
                c.execute("ALTER TABLE jobs ADD COLUMN alignment_json TEXT")
            c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
        else:
            try:
                c.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                for statement in POSTGRES_SCHEMA:
                    c.execute(statement)
            except Exception as exc:
                raise RuntimeError(
                    "Could not initialize PostGIS. Ensure the database has PostGIS installed "
                    "and the configured user can create/use the extension."
                ) from exc


def create_job(job_id: str, width: int, height: int, alignment: dict | None = None):
    with conn() as c:
        c.execute(sql("INSERT INTO jobs(id,status,width,height,created_at,alignment_json) VALUES(?,?,?,?,?,?)"),
                  (job_id, "queued", width, height, time.time(), _json_text(alignment)))


def update_job(job_id: str, **fields):
    allowed = {"status", "error", "width", "height", "n_buildings", "model_mode"}
    if not fields:
        return
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported job fields: {', '.join(sorted(unknown))}")
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(sql(f"UPDATE jobs SET {keys} WHERE id=?"), (*fields.values(), job_id))


def get_job(job_id: str):
    with conn() as c:
        row = c.execute(sql("SELECT * FROM jobs WHERE id=?"), (job_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["alignment"] = json.loads(result.pop("alignment_json")) if result.get("alignment_json") else None
    return result


def save_buildings(job_id: str, rows: list[dict]):
    """Replace all building rows for a job atomically."""
    with conn() as c:
        c.execute(sql("DELETE FROM buildings WHERE job_id=?"), (job_id,))
        for row in rows:
            geometry = _validate_geojson_geometry(row["geometry_geojson"]) if row.get("geometry_geojson") else None
            common = (
                job_id, row["id"], _json_text(row.get("bbox", [])), _json_text(row.get("polygon", [])),
                row.get("damage"), float(row.get("conf") or 0.0), float(row.get("changed_ratio") or 0.0),
                row.get("quadrant"), int(row.get("area", 0)),
                _json_text(row.get("polygons") or [row.get("polygon", [])]),
            )
            if is_postgres():
                c.execute(
                    """INSERT INTO buildings(job_id,id,bbox,polygon,damage,conf,changed_ratio,quadrant,area,polygons,geom)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                         CASE WHEN %s IS NULL THEN NULL ELSE ST_SetSRID(ST_MakeValid(ST_GeomFromGeoJSON(%s)),4326) END)""",
                    (*common, geometry, geometry),
                )
            else:
                c.execute(
                    """INSERT INTO buildings(job_id,id,bbox,polygon,damage,conf,changed_ratio,quadrant,area,polygons,geometry_geojson)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (*common, geometry),
                )


def get_buildings(job_id: str) -> list[dict]:
    if is_postgres():
        query = """SELECT job_id,id,bbox,polygon,damage,conf,changed_ratio,quadrant,area,polygons,
                   ST_AsGeoJSON(geom) AS geometry_geojson FROM buildings WHERE job_id=? ORDER BY id"""
    else:
        query = """SELECT job_id,id,bbox,polygon,damage,conf,changed_ratio,quadrant,area,polygons,geometry_geojson
                   FROM buildings WHERE job_id=? ORDER BY id"""
    with conn() as c:
        records = c.execute(sql(query), (job_id,)).fetchall()
    out = []
    for record in records:
        item = dict(record)
        item.pop("job_id", None)
        for key in ("bbox", "polygon", "polygons", "geometry_geojson"):
            value = item.get(key)
            if isinstance(value, str) and value:
                item[key] = json.loads(value)
        if not item.get("polygons"):
            item["polygons"] = [item.get("polygon") or []]
        out.append(item)
    return out


def find_buildings_intersecting(geojson: dict, job_id: str | None = None) -> list[dict]:
    """Find buildings intersecting EPSG:4326 GeoJSON. Requires PostGIS."""
    if not is_postgres():
        raise RuntimeError("Spatial intersection queries require DATABASE_BACKEND=postgres and PostGIS")
    payload = _validate_geojson_geometry(geojson)
    query = """WITH query_geom AS (
                 SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s),4326) AS geom
               )
               SELECT b.job_id,b.id,b.damage,b.conf,b.quadrant,ST_AsGeoJSON(b.geom) AS geometry_geojson
               FROM buildings b CROSS JOIN query_geom q
               WHERE b.geom IS NOT NULL AND ST_IsValid(q.geom) AND ST_Intersects(b.geom,q.geom)"""
    params: list = [payload]
    if job_id:
        query += " AND b.job_id=%s"
        params.append(job_id)
    query += " ORDER BY b.job_id,b.id"
    with conn() as c:
        validity = c.execute("SELECT ST_IsValid(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)), ST_IsValidReason(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326))", (payload, payload)).fetchone()
        valid = validity[0] if not isinstance(validity, dict) else validity["st_isvalid"]
        reason = validity[1] if not isinstance(validity, dict) else validity["st_isvalidreason"]
        if not valid:
            raise ValueError(f"Invalid GeoJSON geometry: {reason}")
        rows = c.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        if item.get("geometry_geojson"):
            item["geometry_geojson"] = json.loads(item["geometry_geojson"])
        result.append(item)
    return result


def delete_expired_jobs(cutoff: float) -> list[str]:
    """Delete expired jobs and their buildings; return IDs for filesystem cleanup."""
    with conn() as c:
        rows = c.execute(sql("SELECT id FROM jobs WHERE created_at < ?"), (cutoff,)).fetchall()
        ids = [row["id"] if isinstance(row, dict) else row[0] for row in rows]
        if ids:
            # Explicit child deletion works for old SQLite databases without FK constraints.
            if is_postgres():
                c.execute("DELETE FROM jobs WHERE id = ANY(%s)", (ids,))
            else:
                c.executemany("DELETE FROM buildings WHERE job_id=?", [(item,) for item in ids])
                c.executemany("DELETE FROM jobs WHERE id=?", [(item,) for item in ids])
    return ids


def ping() -> dict:
    with conn() as c:
        c.execute("SELECT 1").fetchone()
        if is_postgres():
            version = c.execute("SELECT PostGIS_Version() AS version").fetchone()
            return {"connected": True, "postgis": version["version"] if isinstance(version, dict) else version[0]}
    return {"connected": True, "postgis": None}


def mark_stale_jobs(cutoff: float) -> int:
    """Fail queued/running jobs whose worker heartbeat window has expired."""
    with conn() as c:
        cursor = c.execute(
            sql("UPDATE jobs SET status='failed', error='Job exceeded the configured stale-job window' "
                "WHERE created_at < ? AND status IN ('queued','running')"),
            (cutoff,),
        )
        return cursor.rowcount
