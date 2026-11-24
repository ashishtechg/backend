# Hello Agent API

FastAPI backend for the Hello Agent CSV FAQ app. It accepts one or more CSVs,
loads them into Pandas dataframes, and uses a LangChain agent to answer
questions from matching rows only. A small, safe search tool is used instead
of a Python execution tool.

## Run locally

From this directory, create a virtual environment, install dependencies, and
copy `.env.example` to `.env`. Add an OpenRouter or OpenAI API key to `.env`.
The backend also checks the workspace root `.env` for an existing key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The health endpoint is `http://localhost:8000/health`; interactive API docs
are at `http://localhost:8000/docs`.

## API

`POST /api/ask` accepts multipart form fields:

- `question`: the user's question.
- `files`: zero or more CSV file uploads (select files with the Swagger UI file
  picker; up to 10 files, 5 MB and 10,000 rows each).
- `include_samples`: optionally include the four CSV examples in `data/`.

An uploaded CSV replaces a sample file with the same filename. The response
includes the answer, dataset filenames, and total row count. The backend never
executes code from a question or CSV cell.

If Swagger UI does not show a file picker, restart the backend to load the
custom multipart OpenAPI schema, then refresh `/docs`.

Run the unit tests with:

```powershell
python -m unittest discover -s tests
```
