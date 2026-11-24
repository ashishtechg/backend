import unittest

import pandas as pd

from unittest.mock import patch

from app.agent import (
    NO_MATCH_ANSWER,
    answer_question,
    create_csv_agent,
    find_relevant_rows,
)
from app.csv_data import combine_datasets, load_sample_datasets, read_csv_bytes


class CsvAgentDataTests(unittest.TestCase):
    def test_finds_rows_from_multiple_dataframes(self) -> None:
        datasets = {
            "shipping.csv": pd.DataFrame(
                [{"Category": "Shipping", "Answer": "Standard delivery takes 3 days."}]
            ),
            "billing.csv": pd.DataFrame(
                [{"Category": "Fees", "Answer": "The annual fee is $25."}]
            ),
        }

        rows = find_relevant_rows("How long does standard shipping take?", datasets)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], "shipping.csv")
        self.assertEqual(rows[0]["values"]["Answer"], "Standard delivery takes 3 days.")

    def test_returns_no_rows_for_unmatched_question(self) -> None:
        datasets = {
            "shipping.csv": pd.DataFrame(
                [{"Category": "Shipping", "Answer": "Standard delivery takes 3 days."}]
            )
        }

        self.assertEqual(
            find_relevant_rows("What is the moon made of?", datasets),
            [],
        )
        self.assertEqual(NO_MATCH_ANSWER, "I could not find this information in the uploaded files.")

    def test_unmatched_question_does_not_create_an_llm(self) -> None:
        with patch("app.agent._create_chat_model") as create_model:
            answer = answer_question(
                "What is the moon made of?",
                {"shipping.csv": pd.DataFrame([{"Category": "Shipping"}])},
            )

        self.assertEqual(answer, NO_MATCH_ANSWER)
        create_model.assert_not_called()

    def test_builds_a_langchain_agent_without_a_network_call(self) -> None:
        datasets = {"example.csv": pd.DataFrame([{"Answer": "The answer is 42."}])}
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = create_csv_agent(datasets)

        self.assertTrue(callable(agent.invoke))

    def test_reads_csv_and_rejects_non_csv(self) -> None:
        frame = read_csv_bytes("example.csv", b"Question,Answer\nQ,A\n")
        self.assertEqual(frame.iloc[0]["Answer"], "A")

        with self.assertRaisesRegex(ValueError, "not a CSV"):
            read_csv_bytes("example.txt", b"Question,Answer\nQ,A\n")

    def test_upload_with_matching_name_replaces_sample_dataset(self) -> None:
        sample = {"example.csv": pd.DataFrame([{"Answer": "sample"}])}
        datasets = combine_datasets(
            sample,
            [("example.csv", b"Answer\nuploaded\n")],
        )

        self.assertEqual(datasets["example.csv"].iloc[0]["Answer"], "uploaded")

    def test_loads_all_four_example_datasets(self) -> None:
        datasets = load_sample_datasets()

        self.assertEqual(
            set(datasets),
            {
                "credit_card_terms.csv",
                "ecommerce_faqs.csv",
                "hospital_policy.csv",
                "saas_docs.csv",
            },
        )
        self.assertEqual(sum(len(dataframe) for dataframe in datasets.values()), 60)


if __name__ == "__main__":
    unittest.main()
