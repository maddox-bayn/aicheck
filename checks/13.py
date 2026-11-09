# Exercise 1
import numpy as np

x = matrix([[1, 2]], 2)
assert x.dtype == np.float32 and x.flags.c_contiguous
with raises(ValueError):
    matrix([[np.nan, 2]], 2)

# Exercise 2
import numpy as np

x = np.array([[2, 0], [0, 5]], dtype="float32")
i = faiss_cosine(x, [9, 4])
d, ids = i.search(np.array([[1, 0]], dtype="float32"), 2)
assert ids.tolist() == [[9, 4]] and d.tolist() == [[1, 0]] and x[0, 0] == 2

# Exercise 3
import faiss, numpy as np

i = faiss.IndexFlatIP(2)
i.add(np.array([[1, 0]], dtype="float32"))
assert faiss_query(i, [2, 0], 3) == [(0, 1.0)] and faiss_query(i, [1, 0], 0) == []

# Exercise 4
import chromadb, uuid

c = chromadb.EphemeralClient().create_collection("docs-" + uuid.uuid4().hex)
rows = [dict(id="one", vector=[1.0, 0.0], text="first", tenant="a")]
chroma_write(c, rows)
rows[0]["text"] = "updated"
chroma_write(c, rows)
assert c.count() == 1 and c.get(ids=["one"])["documents"] == ["updated"]

# Exercise 5
from unittest.mock import Mock

c = Mock()
c.query.return_value = {"ids": [["b"]], "distances": [[0.1]]}
assert chroma_search(c, [1.0, 0.0], "T", 1) == [("b", 0.1)]
assert c.query.call_args.kwargs["where"] == {"tenant": {"$eq": "T"}}

# Exercise 6
from unittest.mock import Mock

i = Mock()
assert pinecone_write(i, "A", [{"id": str(x), "values": [x]} for x in range(3)], 2) == 3
assert [len(c.kwargs["vectors"]) for c in i.upsert.call_args_list] == [2, 1]
assert all(c.kwargs["namespace"] == "A" for c in i.upsert.call_args_list)

# Exercise 7
from unittest.mock import Mock
from types import SimpleNamespace as NS

i = Mock()
i.query.return_value = NS(matches=[NS(id="a", score=0.9)])
assert pinecone_search(i, "T", "help", [1], 2) == [("a", 0.9)]
assert i.query.call_args.kwargs["filter"] == {"category": {"$eq": "help"}}

# Exercise 8
assert chunk_id("a:b", "c", "1", 0) != chunk_id("a", "b:c", "1", 0)
assert chunk_id("a", "b", "1", 0) != chunk_id("a", "b", "2", 0)

# Exercise 9
assert (
    stale_ids(["c", "a", "b", "b"], ["a", "new"]) == ["b", "c"]
    and stale_ids([], []) == []
)

# Exercise 10
assert ann_recall([[1, 2], [3], []], [[1, 1], [3, 5], []], 2) == 0.75
with raises(ValueError):
    ann_recall([[]], [[]], 1)
