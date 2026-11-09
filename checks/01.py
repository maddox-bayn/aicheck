# Exercise 1
import numpy as np


np.testing.assert_allclose(unit_rows([[3, 4], [0, 0]]), [[0.6, 0.8], [0, 0]])

# Exercise 2
import numpy as np


np.testing.assert_allclose(
    squared_distances([[1, 2], [3, 0]], [[1, 2], [0, 0]]), [[0, 5], [8, 9]]
)

rng = np.random.default_rng(17)
q, index = rng.normal(size=(4, 7)), rng.normal(size=(9, 7))
expected = ((q[:, None, :] - index[None, :, :]) ** 2).sum(axis=-1)
np.testing.assert_allclose(squared_distances(q, index), expected, atol=1e-12)

# Exercise 3
import numpy as np


np.testing.assert_allclose(
    softmax([[2000, 2000], [-2000, -1999]]), [[0.5, 0.5], [0.26894142, 0.73105858]]
)

# Exercise 4
import numpy as np


np.testing.assert_allclose(
    masked_mean([[[1], [np.nan]], [[8], [9]]], [[1, 0], [0, 0]]), [[1], [0]]
)

# Exercise 5
import numpy as np


z = whiten([[1, 2], [2, 1], [5, 6], [4, 0]])
np.testing.assert_allclose(np.cov(z.T), np.eye(2), atol=1e-5)
with raises(ValueError):
    whiten([[1]])

# Exercise 6
import numpy as np


np.testing.assert_allclose(ridge([[1, 1], [1, 1]], [2, 2], 2), [2 / 3, 2 / 3])

# Exercise 7
import numpy as np


np.testing.assert_allclose(reconstruct(np.diag([5, 2]), 1), np.diag([5, 0]))
np.testing.assert_allclose(reconstruct(np.eye(3), 0), np.zeros((3, 3)))

# Exercise 8
import numpy as np


np.testing.assert_allclose(
    quadratic_scores([[2, 3], [1, 1]], [1, 1], [[2, 1], [1, 3]]), [18, 0]
)

# Exercise 9
import numpy as np


np.testing.assert_allclose(scatter_sum([2, 0, 2], [3, 1, 4], 4), [1, 0, 7, 0])
with raises(ValueError):
    scatter_sum([-1], [1], 2)

# Exercise 10
import numpy as np


n, m, s = merge_moments(
    (2, np.array([2.0]), np.array([[2.0]])), (1, np.array([5.0]), np.array([[0.0]]))
)
assert n == 3
np.testing.assert_allclose([m.item(), s.item()], [3, 8])
