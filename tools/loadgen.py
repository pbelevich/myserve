import asyncio, httpx
from os import environ

HOST = environ.get("HOST", "localhost")
PORT = environ.get("PORT", "8000")

async def run_one(i):
    j = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "stream": True,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": f"Tell me about Titanic disaster"}],
    }
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", f"http://{HOST}:{PORT}/v1/chat/completions", json=j) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line.endswith("[DONE]"):
                    break

async def main():
    tasks = [run_one(i) for i in range(100)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())