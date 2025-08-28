# Observability

```bash
docker run -d --name prometheus --network host \
	-v $(pwd)/tools/prometheus.yml:/etc/prometheus/prometheus.yml \
	prom/prometheus

docker run -d --name grafana --network host grafana/grafana

docker run -d --name otel-collector --network host otel/opentelemetry-collector:0.103.0

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MYSERVE_DEVICE=cuda

uvicorn myserve.main:app --host 0.0.0.0 --port 8000 --reload
```