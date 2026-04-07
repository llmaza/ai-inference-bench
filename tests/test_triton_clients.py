import json
import os
import unittest
from pathlib import Path

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from workloads.bert_classifier.triton.client import TritonBertClient


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
SCENARIO_PATH = REPO_ROOT / "benchmark" / "scenarios" / "bert_inputs.jsonl"
TRITON_URL = os.getenv(
    "TRITON_URL",
    "http://127.0.0.1:8000/v2/models/bert_classifier/infer",
)
MAX_LENGTH = 256


def load_sample_message() -> str:
    with SCENARIO_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            message = row.get("message", "").strip()
            if message:
                return message
    raise RuntimeError("No non-empty sample message found")


class TritonClientSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message = load_sample_message()
        cls.client = TritonBertClient(
            triton_url=TRITON_URL,
            model_dir=MODEL_DIR,
            max_length=MAX_LENGTH,
        )

        try:
            health_url = TRITON_URL.rsplit("/v2/models/", 1)[0] + "/v2/health/live"
            response = requests.get(health_url, timeout=5.0)
            response.raise_for_status()
        except Exception as exc:
            raise unittest.SkipTest("Triton is not reachable: %s" % exc)

        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        cls.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        cls.model.eval()

    def test_triton_returns_expected_logits_shape(self):
        logits = self.client.infer_logits([self.message])
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 7)

        probs = self.client.softmax(logits)
        self.assertEqual(probs.shape, logits.shape)
        self.assertAlmostEqual(float(probs[0].sum()), 1.0, places=5)

    def test_triton_prediction_matches_pytorch_top_class(self):
        triton_prediction = self.client.predict([self.message])[0]

        encoded = self.tokenizer(
            [self.message],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            pytorch_pred = int(torch.argmax(outputs.logits, dim=1)[0].item())

        self.assertEqual(triton_prediction["class_id"], pytorch_pred)
        self.assertIn("topic", triton_prediction)
        self.assertGreaterEqual(triton_prediction["confidence"], 0.0)
        self.assertLessEqual(triton_prediction["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
