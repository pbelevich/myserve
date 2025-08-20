from prometheus_client import Counter, Histogram

REQ_TOTAL = Counter("myserve_requests_total", "Total requests submitted", ["model"])
TOKENS_TOTAL = Counter("myserve_tokens_total", "Total tokens generated", ["model"])
TTFT_HIST = Histogram("myserve_ttft_seconds", "Time to first token", ["model"])  # observe seconds