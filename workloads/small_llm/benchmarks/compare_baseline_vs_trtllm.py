from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stat_mean(summary: dict, top_level_key: str, nested_key: str) -> float | None:
    value = summary.get(top_level_key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _summary_success_rate(summary: dict) -> float | None:
    value = summary.get("success_rate")
    if value is not None:
        return value
    return summary.get("success_rate")


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
            "baseline": _stat_mean(baseline, "mean_latency_ms", "latency_ms"),
            "trtllm": _stat_mean(trtllm, "mean_latency_ms", "latency_ms"),
        },
        "generation_mean_ms": {
            "baseline": _stat_mean(baseline, "mean_generation_ms", "generation_ms"),
            "trtllm": _stat_mean(trtllm, "mean_generation_ms", "generation_ms"),
        },
        "tokens_per_sec_mean": {
            "baseline": _stat_mean(baseline, "mean_tokens_per_sec", "tokens_per_sec"),
            "trtllm": _stat_mean(trtllm, "mean_tokens_per_sec", "tokens_per_sec"),
        },
        "peak_gpu_memory_mb_mean": {
            "baseline": _stat_mean(baseline, "peak_gpu_memory_mb", "peak_gpu_memory_mb"),
            "trtllm": _stat_mean(trtllm, "peak_gpu_memory_mb", "peak_gpu_memory_mb"),
        },
        "baseline_success_rate": _summary_success_rate(baseline),
        "trtllm_success_rate": _summary_success_rate(trtllm),
    }
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
