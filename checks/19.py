# Exercise 1
assert PredictRequest(prompt=" hi ").prompt == "hi"
for row in [
    {"prompt": " "},
    {"prompt": "x", "max_tokens": 513},
    {"prompt": "x", "max_tokens": "1"},
]:
    with raises(ValueError):
        PredictRequest(**row)

# Exercise 2
from fastapi.testclient import TestClient


async def predict(text):
    return text.upper()


with TestClient(prediction_app(predict)) as c:
    assert c.post("/predict", json={"prompt": "hi"}).json() == {"text": "HI"}
    assert c.post("/predict", json={"prompt": ""}).status_code == 422

# Exercise 3
from fastapi.testclient import TestClient

events = []
app = lifespan_app(lambda: events.append("load") or "model", lambda m: events.append(m))
with TestClient(app):
    assert app.state.model == "model" and events == ["load"]
assert events == ["load", "model"]

# Exercise 4
import asyncio, threading

owner = threading.get_ident()
assert asyncio.run(offload(lambda p: (p, threading.get_ident() != owner), "hello")) == (
    "hello",
    True,
)

# Exercise 5
import asyncio
from fastapi import HTTPException


async def case():
    sem = asyncio.Semaphore(1)
    assert await admitted(sem, 1, lambda: asyncio.sleep(0, result="ok")) == "ok"
    assert not sem.locked()
    try:
        await admitted(asyncio.Semaphore(0), 0.01, None)
    except HTTPException as e:
        assert e.status_code == 503
    else:
        raise AssertionError("overload accepted")


asyncio.run(case())

# Exercise 6
import asyncio, json


async def tokens():
    yield "ɛ\nnext"


async def case():
    return [v async for v in sse(tokens())]


frames = asyncio.run(case())
assert json.loads(frames[0].decode()[6:]) == {"token": "ɛ\nnext"} and frames[
    -1
].startswith(b"event: done")

# Exercise 7
from fastapi.testclient import TestClient


async def events():
    yield b"data: ok\n\n"


with TestClient(stream_app(events)) as c:
    r = c.get("/stream")
    assert r.text == "data: ok\n\n" and r.headers["content-type"].startswith(
        "text/event-stream"
    )

# Exercise 8
import asyncio


async def case():
    q = asyncio.Queue()
    for item in ["bad", "good", None]:
        q.put_nowait(item)
    seen = []

    async def process(i):
        if i == "bad":
            raise ValueError()
        seen.append(i)

    async def failed(i, e):
        seen.append(("failed", i))

    await worker(q, process, failed)
    await asyncio.wait_for(q.join(), 1)
    assert seen == [("failed", "bad"), "good"]


asyncio.run(case())

# Exercise 9
import asyncio

seen = []


async def app(scope, receive, send):
    for chunk in [b"a", b"bc"]:
        await send({"type": "http.response.body", "body": chunk})


async def send(msg):
    seen.append(msg["body"])


asyncio.run(ByteMeter(app, seen.append)({"type": "http"}, None, send))
assert seen == [b"a", b"bc", 3]

# Exercise 10
from fastapi.testclient import TestClient

state = [False]
with TestClient(health_app(lambda: state[0])) as c:
    assert c.get("/live").status_code == 200 and c.get("/ready").status_code == 503
    state[0] = True
    assert c.get("/ready").json() == {"ready": True}
