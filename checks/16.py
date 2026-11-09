# Exercise 1
from typing import get_type_hints, get_args
from langchain_core.messages import HumanMessage

merge = get_args(get_type_hints(message_state(), include_extras=True)["messages"])[1]
out = merge(
    [HumanMessage(content="old", id="1")], [HumanMessage(content="new", id="1")]
)
assert len(out) == 1 and out[0].content == "new"

# Exercise 2
assert [route(x) for x in ["research", "draft", "done", "execute"]] == [
    "researcher",
    "writer",
    "finish",
    "review",
]

# Exercise 3
g = conditional_graph(lambda s: {"output": "run"}, lambda s: {"output": "hold"})
assert (
    g.invoke({"approved": True})["output"] == "run"
    and g.invoke({"approved": False})["output"] == "hold"
)

# Exercise 4
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.replay import ReplayChatCompletionClient


async def team_case():
    model = ReplayChatCompletionClient(
        ["draft: APPROVED", "APPROVED", "first", "revise"]
    )
    try:

        def agents():
            return [
                AssistantAgent(name, model_client=model)
                for name in ("writer", "critic")
            ]

        result = await review_team(*agents(), "write a report", 4)
        assert [msg.source for msg in result.messages] == ["user", "writer", "critic"]
        result = await review_team(*agents(), "write again", 2)
        assert len(result.messages) == 3 and "turn" in result.stop_reason.lower()
    finally:
        await model.close()


asyncio.run(team_case())

# Exercise 5
from unittest.mock import patch

with patch("langgraph.types.interrupt", return_value="yes") as pause:
    assert approval_node({"action": "publish"}) == {"approved": False}
    pause.assert_called_once_with({"action": "publish"})

# Exercise 6
from unittest.mock import Mock

b = Mock()
g, c = checkpointed(b, "saver", "run1")
b.compile.assert_called_once_with(checkpointer="saver")
assert c == {"configurable": {"thread_id": "run1"}}

# Exercise 7
from unittest.mock import Mock

g = Mock()
resume(g, "T", False)
cmd, config = g.invoke.call_args.args
assert cmd.resume is False and config["configurable"]["thread_id"] == "T"
with raises(ValueError):
    resume(g, "T", "false")

# Exercise 8
out = fanout([("b", "read"), ("a", "write")])
assert [(m.node, m.arg["id"]) for m in out] == [("worker", "b"), ("worker", "a")]
with raises(ValueError):
    fanout([("a", "x"), ("a", "y")])

# Exercise 9
assert list(merge_results({"b": 2}, {"a": 1, "b": 2}).items()) == [("a", 1), ("b", 2)]
with raises(ValueError):
    merge_results({"a": 1}, {"a": 2})

# Exercise 10
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

calls = []
g = reviewed_graph(lambda x: calls.append(x) or "done", InMemorySaver())
for tid, decision in [("deny", False), ("allow", True)]:
    cfg = {"configurable": {"thread_id": tid}}
    state = g.invoke({"request": "publish"}, cfg)
    assert state["__interrupt__"] and calls == []
    g.invoke(Command(resume=decision), cfg)
assert calls == ["publish"]
