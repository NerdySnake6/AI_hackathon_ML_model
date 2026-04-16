"""FastAPI application for interactive query labeling."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.db.session import DATABASE_URL
from app.pipeline import QueryLabelingPipeline
from app.schemas import BatchQueryInput, LabelResult, QueryInput
from app.settings import DEFAULT_CATALOG_PATH

app = FastAPI(title="ML Query Labeler MVP", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_pipeline() -> QueryLabelingPipeline:
    """Build the default pipeline from CSV or from the configured database."""
    catalog_source = os.getenv("QUERY_LABELER_CATALOG_SOURCE", "csv").strip().lower()
    if catalog_source == "db":
        database_url = os.getenv("QUERY_LABELER_DATABASE_URL", DATABASE_URL)
        return QueryLabelingPipeline.from_db(database_url=database_url)

    catalog_path = os.getenv("QUERY_LABELER_CATALOG_PATH", DEFAULT_CATALOG_PATH)
    return QueryLabelingPipeline.from_csv(catalog_path)


pipeline = build_pipeline()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def demo_page() -> FileResponse:
    """Return the browser demo interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/label", response_model=LabelResult)
def label_query(query: QueryInput) -> LabelResult:
    """Label one search query."""
    try:
        return pipeline.label(query)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/label_batch", response_model=list[LabelResult])
def label_batch(batch: BatchQueryInput) -> list[LabelResult]:
    """Label a bounded batch of search queries."""
    return [pipeline.label(query) for query in batch.queries]
