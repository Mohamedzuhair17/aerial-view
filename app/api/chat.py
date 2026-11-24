from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request

from app import db
from app.llm.agent import answer_question
from app.schemas import ChatIn, ChatOut
from app.storage import job_directory

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, request: Request):
    try:
        job_directory(body.job_id)
    except ValueError:
        raise HTTPException(404, "job not found")
    job = db.get_job(body.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    res = answer_question(body.question, db.get_buildings(body.job_id), request.app.state.llm)
    ids = res["highlight_ids"]
    res["overlay_url"] = f"/jobs/{body.job_id}/overlay?highlight={quote(','.join(ids))}" if 0 < len(ids) <= 150 else None
    return res
