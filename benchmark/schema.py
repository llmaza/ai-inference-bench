from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class BenchmarkRequestResult:
    run_id: str
    timestamp: str
    workload: str
    serving_mode: str
    model_name: str
    hardware: str
    batch_size: int
    concurrency: int
    request_index: int
    success: bool
    status_code: Optional[int]
    wall_latency_ms: Optional[float]
    api_latency_ms: Optional[float]
    predicted_topic: Optional[str]
    confidence: Optional[float]
    error_type: Optional[str]
    input_topic: Optional[str]
    source_row: Optional[int]
    message_len_chars: int
    message_preview: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkRunSummary:
    run_id: str
    timestamp: str
    workload: str
    serving_mode: str
    model_name: str
    hardware: str
    batch_size: int
    concurrency: int
    total_requests: int
    total_http_requests: int
    success_rate: float
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    throughput_rps: Optional[float]
    notes: str
    endpoint: str
    dataset_path: str
    num_requests: int
    timeout_sec: float
    warmup_requests: int = 0
    mean_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    api_p50_ms: Optional[float] = None
    api_p95_ms: Optional[float] = None
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    successful_requests: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
