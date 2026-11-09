# Exercise 1
assert (
    chunks("abcdef", 4, 2) == [(0, 4, "abcd"), (2, 6, "cdef")]
    and chunks("", 2, 0) == []
)

# Exercise 2
assert pack_sentences(["ab", "cd", "ef"], len, 5) == ["ab cd", "ef"]
with raises(ValueError):
    pack_sentences(["large"], len, 2)

# Exercise 3
assert children([("d", "red blue")], str.split)[1] == {
    "parent": "d",
    "ordinal": 1,
    "text": "blue",
}
with raises(ValueError):
    children([("d", "a"), ("d", "b")], str.split)

# Exercise 4
assert hydrate(["gone", "b", "b", "a"], {"a": 1, "b": 2}, 2) == [("b", 2), ("a", 1)]

# Exercise 5
assert fuse([["a", "b", "b"], ["b", "c"]], 0) == ["b", "a", "c"]
assert fuse([["z"], ["a"]]) == ["a", "z"]

# Exercise 6
d = [{"text": "one"}, {"text": "two"}, {"text": "three"}]
assert rerank("q", d, lambda pairs: [0, 2, 2], 2) == d[1:]
with raises(ValueError):
    rerank("q", d, lambda pairs: [1], 2)

# Exercise 7
assert diversify([1, 0.95, 0.8], [[1, 1, 0], [1, 1, 0], [0, 0, 1]], 2, 0.5) == [0, 2]

# Exercise 8
assert context_pack([("a", "x"), ("b", "long"), ("c", "y")], len, 11) == (
    "[a] x\n[c] y",
    ["a", "c"],
)

# Exercise 9
assert answer_or_abstain("q", [], 0.5, None) == {"answer": None, "sources": []}
assert answer_or_abstain("q", [{"id": "d", "score": 0.7}], 0.5, lambda q, h: "yes")[
    "sources"
] == ["d"]

# Exercise 10
loaded = []


def load(i):
    loaded.append(i)
    return i


assert scoped_answer(
    "q", ["secret", "ok", "ok"], {"ok"}, load, lambda q, d: d, lambda q, d: d
) == ["ok"]
assert (
    loaded == ["ok"] and scoped_answer("q", ["secret"], set(), None, None, None) is None
)
