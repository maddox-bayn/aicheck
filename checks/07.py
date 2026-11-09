# Exercise 1
v, x = fit_text(["red red bike", "blue bike"])
assert x.shape == (2, 3) and v.transform(["unseen"]).nnz == 0

# Exercise 2
x = hashed(["red red", "blue"], 16)
assert x.shape == (2, 16) and x.sum(axis=1).A1.tolist() == [2, 1]

# Exercise 3
import numpy as np


np.testing.assert_allclose(
    pool([[[2, 4], [9, 7]], [[0, 1], [1, 2]]], [[1, 0], [0, 0]]), [[2, 4], [0, 0]]
)

# Exercise 4
calls = []


def enc(xs):
    calls.append(len(xs))
    return [[len(x)] for x in xs]


assert embed_batches(["a", "bb", "c"], enc, 2) == [[1], [2], [1]] and calls == [2, 1]
with raises(ValueError):
    embed_batches(["x"], lambda xs: [], 1)

# Exercise 5
assert nearest([1, 0], [[1, 0], [0, 0], [2, 0]], 3).tolist() == [0, 2, 1]

# Exercise 6
p, z = project([[0, 0], [2, 0], [4, 0]], [[6, 0]], 1)
assert z.shape == (1, 1) and abs(abs(z.item()) - 4) < 1e-8

# Exercise 7
assert dimension([0.6, 0.3, 0.1], 0.9) == 2
with raises(ValueError):
    dimension([0.5], 0.9)

# Exercise 8
from scipy.sparse import csr_matrix

model, z = lsa(csr_matrix([[1, 0, 2], [0, 3, 0], [2, 0, 1]]), 2)
assert z.shape == (3, 2) and model.components_.shape == (2, 3)

# Exercise 9
import numpy as np


x = np.arange(24).reshape(8, 3)
z = tsne_map(x, 2)
assert z.shape == (8, 2) and np.isfinite(z).all()
with raises(ValueError):
    tsne_map(x, 8)

# Exercise 10
import numpy as np

x = np.arange(10.0).reshape(-1, 1)
assert neighborhood_quality(x, x, 2) == 1
with raises(ValueError):
    neighborhood_quality(x, x, 5)
