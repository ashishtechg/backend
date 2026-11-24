from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import pandas as pd

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ROWS_PER_FILE = 10_000
MAX_FILES_PER_REQUEST = 10


def read_csv_bytes(filename: str, content: bytes) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"{filename} is not a CSV file.")
    if not content:
        raise ValueError(f"{filename} is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"{filename} exceeds the 5 MB file limit.")

    try:
        dataframe = pd.read_csv(
            BytesIO(content),
            encoding="utf-8-sig",
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not read {filename} as a valid UTF-8 CSV.") from error

    if dataframe.empty:
        raise ValueError(f"{filename} has no data rows.")
    if len(dataframe) > MAX_ROWS_PER_FILE:
        raise ValueError(f"{filename} exceeds the {MAX_ROWS_PER_FILE:,}-row limit.")
    return dataframe


def load_sample_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for path in sorted(DATA_DIRECTORY.glob("*.csv")):
        datasets[path.name] = read_csv_bytes(path.name, path.read_bytes())
    return datasets


def combine_datasets(
    sample_datasets: dict[str, pd.DataFrame],
    uploads: Iterable[tuple[str, bytes]],
) -> dict[str, pd.DataFrame]:
    uploaded_files = list(uploads)
    if len(uploaded_files) > MAX_FILES_PER_REQUEST:
        raise ValueError(f"Upload no more than {MAX_FILES_PER_REQUEST} CSV files.")

    datasets = dict(sample_datasets)
    for filename, content in uploaded_files:
        safe_filename = Path(filename.replace("\\", "/")).name
        if not safe_filename:
            raise ValueError("Every uploaded file must have a filename.")
        datasets[safe_filename] = read_csv_bytes(safe_filename, content)
    return datasets
