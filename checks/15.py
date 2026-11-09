# Exercise 1
r = {}
register(r, "lookup", dict, str)
assert r["lookup"] == (dict, str)
with raises(ValueError):
    register(r, "lookup", dict, str)

# Exercise 2
from pydantic import BaseModel, ConfigDict


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int


r = {"double": (Args, lambda x: 2 * x)}
assert dispatch(r, "double", '{"x":2}') == 4
with raises(ValueError):
    dispatch(r, "double", '{"x":"2"}')

# Exercise 3
assert authorized("read", {"read": ["r"]}, {"r"}, lambda: 42) == 42
with raises(PermissionError):
    authorized("write", {}, set(), None)

# Exercise 4
import asyncio

assert asyncio.run(timed_call(asyncio.sleep, {"delay": 0, "result": "ok"}, 1)) == "ok"
with raises(TimeoutError):
    asyncio.run(timed_call(asyncio.sleep, {"delay": 1}, 0.01))

# Exercise 5
import asyncio

state = {"active": 0, "peak": 0}


async def job(i):
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    await asyncio.sleep(0.001)
    state["active"] -= 1
    return i


assert (
    asyncio.run(parallel_tools([lambda i=i: job(i) for i in range(5)], 2))
    == list(range(5))
    and state["peak"] == 2
)

# Exercise 6
assert bounded_json("ɛ", 4) == '"ɛ"'
with raises(ValueError):
    bounded_json("ɛ", 3)

# Exercise 7
a = ("x", {"a": 1, "b": 2})
assert looping([a, ("x", {"b": 2, "a": 1})], 2) and not looping([a], 2)

# Exercise 8
c = {}
assert (
    replay(c, "1", "read", {}, lambda: 3) == 3 and replay(c, "1", "read", {}, None) == 3
)
with raises(ValueError):
    replay(c, "1", "write", {}, None)

# Exercise 9
from unittest.mock import Mock

assert tool_result(Mock(side_effect=ValueError("sensitive"))) == {
    "ok": False,
    "error": "invalid_arguments",
}
with raises(KeyError):
    tool_result(Mock(side_effect=KeyError()))

# Exercise 10
events = []
times = iter([3, 5])
assert (
    audited("x", "read", lambda: "secret", lambda: next(times), events.append)
    == "secret"
)
assert events == [{"call_id": "x", "tool": "read", "status": "ok", "elapsed": 2}]
