# Exercise 1
from unittest.mock import Mock
from types import SimpleNamespace as NS

c = Mock()
c.chat.completions.create.return_value = NS(choices=[NS(message="ok")])
assert complete(c, "local", []) == "ok"
c.chat.completions.create.assert_called_once_with(
    model="local", messages=[], timeout=30
)

import httpx
import json
from openai import OpenAI


def transport(request):
    body = json.loads(request.content)
    assert body["model"] == "tiny" and body["messages"][0]["content"] == "hello"
    return httpx.Response(
        200,
        json={
            "id": "fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "tiny",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "world"},
                }
            ],
        },
    )


with OpenAI(
    api_key="offline-fixture",
    base_url="http://fixture.invalid/v1",
    http_client=httpx.Client(transport=httpx.MockTransport(transport)),
) as client:
    message = complete(client, "tiny", [{"role": "user", "content": "hello"}])
    assert message.content == "world"

# Exercise 2
h = [{"role": "system", "content": "brief"}]
assert chat_turn(h, "hi", lambda m: "hey")[-1]["content"] == "hey" and len(h) == 1
with raises(ValueError):
    chat_turn(h, "hi", lambda m: None)

# Exercise 3
assert trim_turns("s", [("aa", "b"), ("c", "d")], len, 4) == ["s", "c", "d"]
with raises(ValueError):
    trim_turns("xx", [], len, 1)

# Exercise 4
from types import SimpleNamespace as NS


class Stream(list):
    def close(self):
        self.closed = True


s = Stream(
    [
        NS(choices=[]),
        NS(choices=[NS(delta=NS(content=None))]),
        NS(choices=[NS(delta=NS(content="hi"))]),
    ]
)
assert list(text_deltas(s)) == ["hi"] and s.closed

# Exercise 5
assert tool_arguments([(1, '{"b":'), (0, "{}"), (1, "2}")]) == [{}, {"b": 2}]
with raises(ValueError):
    tool_arguments([(0, "[]")])

# Exercise 6
from pydantic import BaseModel, ConfigDict


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


t = function_tool("weather", "Get weather", Args)
assert t["function"]["strict"] and t["function"]["parameters"]["required"] == ["city"]

# Exercise 7
assert tool_message("t1", {"b": 1, "a": 2}) == {
    "role": "tool",
    "tool_call_id": "t1",
    "content": '{"a": 2, "b": 1}',
}
with raises(ValueError):
    tool_message("x", float("nan"))

# Exercise 8
from unittest.mock import Mock

call = Mock(side_effect=[TimeoutError(), TimeoutError(), "ok"])
delays = []
assert retry_read(call, TimeoutError, delays.append) == "ok" and delays == [1, 2]
with raises(KeyError):
    retry_read(Mock(side_effect=KeyError()), TimeoutError, delays.append)

# Exercise 9
from decimal import Decimal

assert usage_cost(1000, 200, "2.5", "10") == Decimal("0.0045")
with raises(ValueError):
    usage_cost(-1, 0, 1, 1)

# Exercise 10
replies = iter(
    [
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "assistant", "content": "done"},
    ]
)
out = tool_chat([], lambda m: next(replies), lambda c: "42", 2)
assert out[1]["tool_call_id"] == "a" and out[-1]["content"] == "done"
with raises(RuntimeError):
    tool_chat([], lambda m: {"tool_calls": [{"id": "x"}]}, lambda c: "", 1)
