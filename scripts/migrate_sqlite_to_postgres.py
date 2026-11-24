"""Copy existing SQLite jobs/building records into the configured PostgreSQL/PostGIS DB.

Usage:
  DATABASE_BACKEND=postgres DATABASE_URL=postgresql://... \
    python scripts/migrate_sqlite_to_postgres.py --sqlite data/app.db

This copies pixel-space geometry as JSON and only writes PostGIS geom when the
source row already contains explicit EPSG:4326 GeoJSON in geometry_geojson.
"""
import argparse
import json
import sqlite3
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default="data/app.db", help="Path to source SQLite database")
    parser.add_argument("--jobs-dir", default="data/jobs", help="Source directory containing stored job images")
    parser.add_argument("--skip-files", action="store_true", help="Migrate database rows only; do not copy job images")
    args = parser.parse_args()
    if settings.database_backend.lower() not in {"postgres", "postgresql", "postgis"}:
        raise SystemExit("Set DATABASE_BACKEND=postgres and DATABASE_URL before running migration")
    source_path = Path(args.sqlite)
    if not source_path.exists():
        raise SystemExit(f"SQLite database not found: {source_path}")

    db.init_db()
    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    try:
        jobs = source.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        job_cols = {r["name"] for r in source.execute("PRAGMA table_info(jobs)").fetchall()}
        bcols = {r["name"] for r in source.execute("PRAGMA table_info(buildings)").fetchall()}
        migrated = 0
        copied_files = 0
        source_jobs_dir = Path(args.jobs_dir)
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            with db.conn() as target:
                alignment_json = job["alignment_json"] if "alignment_json" in job_cols else None
                target.execute(
                    """INSERT INTO jobs(id,status,error,width,height,n_buildings,model_mode,created_at,alignment_json)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status,error=EXCLUDED.error,
                         width=EXCLUDED.width,height=EXCLUDED.height,n_buildings=EXCLUDED.n_buildings,
                         model_mode=EXCLUDED.model_mode,created_at=EXCLUDED.created_at,
                         alignment_json=EXCLUDED.alignment_json""",
                    tuple(job[k] for k in ("id", "status", "error", "width", "height", "n_buildings", "model_mode", "created_at")) + (alignment_json,),
                )
            select_cols = ["id", "bbox", "polygon", "damage", "conf", "changed_ratio", "quadrant", "area"]
            if "polygons" in bcols:
                select_cols.append("polygons")
            if "geometry_geojson" in bcols:
                select_cols.append("geometry_geojson")
            records = source.execute(
                f"SELECT {','.join(select_cols)} FROM buildings WHERE job_id=? ORDER BY id", (job["id"],)
            ).fetchall()
            rows = []
            for rec in records:
                d = dict(rec)
                d["bbox"] = json.loads(d["bbox"]) if d["bbox"] else []
                d["polygon"] = json.loads(d["polygon"]) if d["polygon"] else []
                d["polygons"] = json.loads(d["polygons"]) if d.get("polygons") else [d["polygon"]]
                if d.get("geometry_geojson"):
                    d["geometry_geojson"] = json.loads(d["geometry_geojson"])
                rows.append(d)
            # Also replace with an empty set: this removes stale target rows when
            # the source job legitimately contains no detections.
            db.save_buildings(job["id"], rows)
            if not args.skip_files:
                # Job IDs are generated hex strings by this application; reject any
                # unexpected path-like values before copying persisted artifacts.
                job_id = str(job["id"])
                if not (12 <= len(job_id) <= 32) or any(ch not in "0123456789abcdef" for ch in job_id):
                    raise ValueError(f"Unsafe job ID in source database: {job_id!r}")
                source_dir = source_jobs_dir / job_id
                if source_dir.is_dir():
                    target_dir = settings.jobs_dir / job_id
                    if source_dir.resolve() != target_dir.resolve():
                        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                    copied_files += 1
            migrated += 1
        print(f"Migration complete: {migrated} jobs copied from {source_path}")
        if not args.skip_files:
            print(f"Copied image/artifact folders for {copied_files} jobs from {source_jobs_dir}")
            if copied_files < migrated:
                print("WARNING: some source job folders were not found; their images/overlays will not be available until restored.")
        print("Pixel polygons remain pixel coordinates; only explicit EPSG:4326 GeoJSON populates PostGIS geom.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
