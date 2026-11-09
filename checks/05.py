# Exercise 1
import numpy as np


loss, g = mse_gradient([[1], [2]], np.array([1, 2]), [0])
np.testing.assert_allclose([loss, g[0]], [2.5, -5])

# Exercise 2
assert logistic_loss([1000, -1000], [1, 0]) < 1e-8
assert logistic_loss([-1000], [1]) == 1000

# Exercise 3
import numpy as np


v, g = huber([-3, 0.5], 1)
np.testing.assert_allclose(v, [2.5, 0.125])
np.testing.assert_allclose(g, [-1, 0.5])

# Exercise 4
import numpy as np


w = np.array([2.0, -3.0])
np.testing.assert_allclose(numerical_gradient(lambda p: np.sum(p**3), w), [12, 27])
np.testing.assert_equal(w, [2, -3])

# Exercise 5
import numpy as np


np.testing.assert_allclose(descend(lambda w: 2 * (w - 3), [0], 0.5, 10, 1e-8), [3])

# Exercise 6
import numpy as np


w, v = momentum([2.0], [1.0], np.array([3.0]), 0.1, 0.5)
np.testing.assert_allclose([w.item(), v.item()], [1.65, 3.5])

# Exercise 7
import numpy as np


w, m, v = adam(1.0, -2.0, 0.0, 0.0, 1)
np.testing.assert_allclose([w, m, v], [1.001, -0.2, 0.004])
with raises(ValueError):
    adam(1, 1, 0, 0, 0)

# Exercise 8
import numpy as np


loss, g = multiclass([[1000, 1000]], [1])
np.testing.assert_allclose([loss, *g[0]], [np.log(2), 0.5, -0.5])

# Exercise 9
import numpy as np


x, w, v = np.array([[2.0]]), np.array([[0.5]]), np.array([[3.0]])
a, b = mlp_backward(x, w, v)
np.testing.assert_allclose(
    [a.item(), b.item()], [6 * (1 - np.tanh(1) ** 2), np.tanh(1)]
)

# Exercise 10
import numpy as np


x, s = backtrack(lambda a: (a * a).sum(), np.array([2.0]), np.array([4.0]))
assert s == 0.5 and x.item() == 0
with raises(RuntimeError):
    backtrack(lambda a: 1, np.array([0.0]), np.array([1.0]), trials=2)
