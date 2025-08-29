from prometheus_client import Counter, Histogram, Gauge

# Counters
REQ_TOTAL = Counter("myserve_requests_total", "Total requests submitted", ["model"])
TOKENS_TOTAL = Counter("myserve_tokens_total", "Total tokens generated", ["model"])

# Histograms (seconds)
TTFT_HIST = Histogram("myserve_ttft_seconds", "Time to first token", ["model"], buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5))
QUEUE_WAIT_HIST = Histogram("myserve_queue_wait_seconds", "Ingress → prefill start", ["model"], buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2))
PREFILL_HIST = Histogram("myserve_prefill_seconds", "Prefill batch forward time", ["model"], buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1))
DECODE_STEP_HIST = Histogram("myserve_decode_step_seconds", "Single decode tick forward time", ["model"], buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05))
E2E_HIST = Histogram("myserve_e2e_seconds", "End-to-end request time", ["model"], buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 15, 30, 60))

# Batch size histograms
PREFILL_BATCH_HIST = Histogram("myserve_prefill_batch_size", "Prefill micro-batch size", ["model"], buckets=(1, 2, 4, 8, 16, 32))
DECODE_BATCH_HIST = Histogram("myserve_decode_batch_size", "Decode micro-batch size", ["model"], buckets=(1, 2, 4, 8, 16, 32, 64))

# Gauges
INFLIGHT = Gauge("myserve_inflight_requests", "Active in-progress requests", ["model"])
QUEUE_DEPTH = Gauge("myserve_queue_depth", "Ingress queue depth (NEW) ")
ACTIVE_PREFILL = Gauge("myserve_active_prefill", "Requests included in the last prefill batch", ["model"])
ACTIVE_DECODE = Gauge("myserve_active_decode", "Requests included in the last decode batch", ["model"])
# new memory gauges
MEM_RESERVED_BYTES = Gauge("myserve_mem_reserved_bytes", "Bytes reserved for KV/workspace", ["model"])
MEM_FREE_BYTES = Gauge("myserve_mem_free_bytes", "Approx free device bytes")  # no labels
MODEL_BYTES = Gauge("myserve_model_bytes", "Approx model bytes in memory", ["model"])
