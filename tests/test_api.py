import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class AskEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_openapi_marks_files_as_uploads(self) -> None:
        schema = self.client.get("/openapi.json").json()
        request_schema_ref = schema["paths"]["/api/ask"]["post"]["requestBody"][
            "content"
        ]["multipart/form-data"]["schema"]["$ref"]
        request_schema_name = request_schema_ref.rsplit("/", 1)[-1]
        files_schema = schema["components"]["schemas"][request_schema_name][
            "properties"
        ]["files"]

        self.assertEqual(files_schema["type"], "array")
        self.assertEqual(files_schema["items"], {"type": "string", "format": "binary"})

    def test_asks_about_uploaded_csv(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("app.main.answer_question", return_value="Three business days.") as ask:
                response = self.client.post(
                    "/api/ask",
                    data={
                        "question": "How long is shipping?",
                        "include_samples": "false",
                    },
                    files=[
                        (
                            "files",
                            ("shipping.csv", b"Category,Answer\nShipping,3 days\n", "text/csv"),
                        )
                    ],
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Three business days.")
        self.assertEqual(response.json()["files"], ["shipping.csv"])
        self.assertEqual(response.json()["rows_loaded"], 1)
        ask.assert_called_once()

    def test_rejects_request_without_data(self) -> None:
        response = self.client.post(
            "/api/ask",
            data={"question": "How long is shipping?", "include_samples": "false"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload one or more CSV files", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
