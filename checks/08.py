# Exercise 1
assert byte_ids("ɛ́") == list("ɛ́".encode()) and byte_ids("") == []

# Exercise 2
assert pair_counts({("a", "a", "a"): 3, ("a", "b"): 2}) == {
    ("a", "a"): 6,
    ("a", "b"): 2,
}

# Exercise 3
assert (
    merge_pair(list("aaa"), ("a", "a")) == ["aa", "a"]
    and merge_pair([], ("x", "y")) == []
)

# Exercise 4
assert train_merges({"ab": 2, "ac": 2}, 2) == [("a", "b"), ("a", "c")]
assert train_merges({"x": 3}, 2) == []

# Exercise 5
t = hf_bpe(["hello world", "hello there"], 32)
assert t.encode("hello").ids and t.encode("🚀").tokens == ["[UNK]"]

# Exercise 6
from types import SimpleNamespace as NS

t = NS(encode=lambda s: NS(ids=[99, 5, 4], offsets=[(0, 0), (0, 1), (2, 3)]))
assert offset_audit(t, "ɛ x") == [(5, "ɛ"), (4, "x")]

# Exercise 7
import tiktoken

enc = tiktoken.Encoding(
    name="bytes",
    pat_str=r"(?s).",
    mergeable_ranks={bytes([i]): i for i in range(256)},
    special_tokens={"<X>": 256},
)
assert token_cost(enc, ["", "ɛ", "<X>"]) == [0, 2, 3]

# Exercise 8
assert fit_prefix("aɛb", lambda s: s.encode(), 3) == "aɛ"
assert fit_prefix("abc", lambda s: [1] if s == "abc" else list(s), 1) == "abc"

# Exercise 9
assert token_windows([1, 2, 3, 4, 5], 3, 1) == [[1, 2, 3], [3, 4, 5]]
with raises(ValueError):
    token_windows([], 2, 2)

# Exercise 10
b = causal_batch([[0, 2], [0]], 0)
assert b == {
    "input_ids": [[0, 2], [0, 0]],
    "attention_mask": [[1, 1], [1, 0]],
    "labels": [[0, 2], [0, -100]],
}
