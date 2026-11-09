# Exercise 1
assert exact_match("  CAFÉ ", "cafe\u0301") and not exact_match("café", "cafe")

# Exercise 2
assert (
    token_f1("x x y", "x z") == 0.4 and token_f1("", "") == 1 and token_f1("x", "") == 0
)

# Exercise 3
assert semantic_score("a", "b", lambda x: [[1, 0], [-2, 0]]) == -1
with raises(ValueError):
    semantic_score("a", "b", lambda x: [[0], [1]])

# Exercise 4
assert mrr([["x", "a"], ["z"]], [{"a"}, {"b"}], 2) == 0.25

# Exercise 5
s = ragas_sample("Q", ["source"], "A", "gold")
assert (
    s.user_input == "Q"
    and s.retrieved_contexts == ["source"]
    and s.reference == "gold"
    and s.response == "A"
)

# Exercise 6
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from ragas.metrics.collections import ExactMatch

metric = ExactMatch()
assert asyncio.run(ragas_score(metric, {"reference": "A", "response": "A"})) == 1
assert asyncio.run(ragas_score(metric, {"reference": "A", "response": "B"})) == 0
bad = SimpleNamespace(
    ascore=AsyncMock(return_value=SimpleNamespace(value=float("nan")))
)
with raises(ValueError):
    asyncio.run(ragas_score(bad, {}))

# Exercise 7
from unittest.mock import Mock

m = Mock(score=0.8)
m.is_successful.return_value = True
assert deepeval_case(m, "Q", "A", "B", ["s"]) == {"score": 0.8, "passed": True}
case = m.measure.call_args.args[0]
assert case.actual_output == "A" and case.retrieval_context == ["s"]

# Exercise 8
assert (
    judge_answer(
        "q", "a", "accuracy", lambda r, d: '{"score":4,"rationale":"grounded"}'
    )["score"]
    == 4
)
with raises(ValueError):
    judge_answer("q", "a", "accuracy", lambda r, d: '{"score":9,"rationale":"x"}')

# Exercise 9
assert (
    pairwise("best", "worse", lambda a, b: "first" if a == "best" else "second") == "a"
)
assert pairwise("a", "b", lambda a, b: "first") == "tie"

# Exercise 10
assert paired_interval([1, 2], [2, 3], 25) == (1, 1, 1)
with raises(ValueError):
    paired_interval([1], [], 25)
