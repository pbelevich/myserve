from prometheus_client import Counter, Histogram, Gauge

REQ_TOTAL = Counter("myserve_requests_total", "Total requests submitted", ["model"])
TOKENS_TOTAL = Counter("myserve_tokens_total", "Total tokens generated", ["model"])
TTFT_HIST = Histogram("myserve_ttft_seconds", "Time to first token", ["model"])  # observe seconds

# new memory gauges
MEM_RESERVED_BYTES = Gauge("myserve_mem_reserved_bytes", "Bytes reserved for KV/workspace", ["model"])
MEM_FREE_BYTES = Gauge("myserve_mem_free_bytes", "Approx free device bytes")  # no labels
