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
