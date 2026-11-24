import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from starlette.concurrency import run_in_threadpool

from app.agent import answer_question
from app.csv_data import combine_datasets, load_sample_datasets

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="Hello Agent API", version="1.0.0")


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    request_schema_ref = schema["paths"]["/api/ask"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]["$ref"]
    request_schema_name = request_schema_ref.rsplit("/", 1)[-1]
    files_schema = schema["components"]["schemas"][request_schema_name]["properties"][
        "files"
    ]
    files_schema["items"] = {"type": "string", "format": "binary"}

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    include_samples: bool = Form(default=False),
) -> dict[str, object]:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        sample_datasets = load_sample_datasets() if include_samples else {}
        uploads: list[tuple[str, bytes]] = []
        for upload in files:
            filename = upload.filename or ""
            content = await upload.read(5 * 1024 * 1024 + 1)
            uploads.append((filename, content))
        datasets = combine_datasets(sample_datasets, uploads)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if not datasets:
        raise HTTPException(
            status_code=400,
            detail="Upload one or more CSV files or enable the sample datasets.",
        )
    if not (
        os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    ):
        raise HTTPException(
            status_code=503,
            detail="Configure OPENROUTER_API_KEY or OPENAI_API_KEY in the backend environment.",
        )

    row_count = sum(len(dataframe) for dataframe in datasets.values())
    answer = await run_in_threadpool(answer_question, cleaned_question, datasets)
    return {
        "answer": answer,
        "files": list(datasets),
        "rows_loaded": row_count,
    }
