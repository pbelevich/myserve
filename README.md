# MyServe

## Setup
```bash
git clone https://github.com/pbelevich/myserve.git && cd myserve
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Test
```bash
pytest
```

## Run
```bash
uvicorn myserve.main:app --host 0.0.0.0 --port 8000 --reload
```

## Send sample request
```bash
curl -s http://${HOST}:${PORT}/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }' | jq
  ```

## Benchmark

```
pip install sglang
```

```bash
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://${HOST}:${PORT} \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset-name random --num-prompts 10 \
  --random-input 128 --random-output 128
```
