from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage A baseline summary against TRT-LLM direct summary.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--trtllm-summary", required=True)
    args = parser.parse_args()

    baseline = load_summary(Path(args.baseline_summary))
    trtllm = load_summary(Path(args.trtllm_summary))

    comparison = {
        "baseline_run_id": baseline.get("run_id"),
        "trtllm_run_id": trtllm.get("run_id"),
        "latency_mean_ms": {
            "baseline": baseline.get("latency_ms", {}).get("mean"),
            "trtllm": trtllm.get("latency_ms", {}).get("mean"),
        },
        "generation_mean_ms": {
            "baseline": baseline.get("generation_ms", {}).get("mean"),
            "trtllm": trtllm.get("generation_ms", {}).get("mean"),
        },
        "tokens_per_sec_mean": {
            "baseline": baseline.get("tokens_per_sec", {}).get("mean"),
            "trtllm": trtllm.get("tokens_per_sec", {}).get("mean"),
        },
        "peak_gpu_memory_mb_mean": {
            "baseline": baseline.get("peak_gpu_memory_mb", {}).get("mean"),
            "trtllm": trtllm.get("peak_gpu_memory_mb", {}).get("mean"),
        },
    }
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
