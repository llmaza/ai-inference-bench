import math
import statistics
from typing import List, Optional

from benchmark.schema import BenchmarkRunSummary


def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def summarize_run(
    *,
    run_id: str,
    timestamp: str,
    workload: str,
    serving_mode: str,
    model_name: str,
    hardware: str,
    batch_size: int,
    concurrency: int,
    total_requests: int,
    total_http_requests: int,
    notes: str,
    endpoint: str,
    dataset_path: str,
    timeout_sec: float,
    warmup_requests: int,
    total_wall_sec: float,
    results: List[dict],
) -> BenchmarkRunSummary:
    success_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    wall_latencies = sorted(r["wall_latency_ms"] for r in results if r["wall_latency_ms"] is not None)
    api_latencies = sorted(
        r["api_latency_ms"] for r in success_results if r["api_latency_ms"] is not None
    )

    throughput_rps = total_requests / total_wall_sec if total_wall_sec > 0 else None
    success_rate = len(success_results) / total_requests if total_requests else 0.0
    error_rate = len(failed_results) / total_requests if total_requests else 0.0
    timeout_rate = (
        sum(1 for r in failed_results if r["error_type"] == "timeout") / total_requests
        if total_requests
        else 0.0
    )

    return BenchmarkRunSummary(
        run_id=run_id,
        timestamp=timestamp,
        workload=workload,
        serving_mode=serving_mode,
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        concurrency=concurrency,
        total_requests=total_requests,
        total_http_requests=total_http_requests,
        success_rate=round(success_rate, 4),
        p50_ms=round(percentile(wall_latencies, 50), 2) if wall_latencies else None,
        p95_ms=round(percentile(wall_latencies, 95), 2) if wall_latencies else None,
        throughput_rps=round(throughput_rps, 4) if throughput_rps is not None else None,
        notes=notes,
        endpoint=endpoint,
        dataset_path=dataset_path,
        num_requests=total_requests,
        timeout_sec=timeout_sec,
        warmup_requests=warmup_requests,
        mean_ms=round(statistics.mean(wall_latencies), 2) if wall_latencies else None,
        p99_ms=round(percentile(wall_latencies, 99), 2) if wall_latencies else None,
        min_ms=round(min(wall_latencies), 2) if wall_latencies else None,
        max_ms=round(max(wall_latencies), 2) if wall_latencies else None,
        api_p50_ms=round(percentile(api_latencies, 50), 2) if api_latencies else None,
        api_p95_ms=round(percentile(api_latencies, 95), 2) if api_latencies else None,
        error_rate=round(error_rate, 4),
        timeout_rate=round(timeout_rate, 4),
        successful_requests=len(success_results),
    )
