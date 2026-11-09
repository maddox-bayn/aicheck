# Exercise 1
p = classifier().fit([[0], [1], [8], [9]], [0, 0, 1, 1])
assert p.predict([[0], [9]]).tolist() == [0, 1] and p.steps[0][1].mean_[0] == 4.5

# Exercise 2
import numpy as np


t = QuantileClipper(0, 1).fit([[1, 10], [3, 20]])
np.testing.assert_equal(t.transform([[0, 100]]), [[1, 20]])
with raises(ValueError):
    QuantileClipper(0.8, 0.1).fit([[1]])

# Exercise 3
import numpy as np

t = preprocess([0], [1]).fit(np.array([[1, "a"], [3, "b"]], dtype=object))
np.testing.assert_allclose(t.transform(np.array([[2, "c"]], dtype=object)), [[0, 0, 0]])

# Exercise 4
import numpy as np


g = np.array([1, 1, 2, 2, 3, 3])
folds = group_folds(g, 3)
assert len(folds) == 3 and all(set(g[a]).isdisjoint(g[b]) for a, b in folds)
assert sorted(np.concatenate([b for a, b in folds])) == list(range(6))

# Exercise 5
folds = temporal_folds(18, 3, 2)
assert len(folds) == 3 and all(b.min() - a.max() == 3 for a, b in folds)

# Exercise 6
import numpy as np
from sklearn.linear_model import LogisticRegression

scores = oof_scores(
    LogisticRegression(), np.arange(12).reshape(-1, 1), [0] * 6 + [1] * 6, 3
)
assert (
    scores.shape == (12,)
    and min(scores) >= 0
    and max(scores) <= 1
    and scores[0] < scores[-1]
)

# Exercise 7
import numpy as np


assert f1_threshold([0, 1, 1], [0.1, 0.6, 0.8]) == 0.6
assert np.isinf(f1_threshold([0, 0], [0.1, 0.2]))

# Exercise 8
assert recall_audit([1, 1, 0], [1, 0, 0], [0, 1, 2]) == {0: 1.0, 1: 0.5, 2: 0.0}

# Exercise 9
assert abs(calibration_error([0, 1, 1], [0, 0.5, 1], 2) - 1 / 6) < 1e-9
with raises(ValueError):
    calibration_error([1], [2], 3)

# Exercise 10
import numpy as np

s = tune(np.arange(18).reshape(-1, 1), [0] * 9 + [1] * 9, [0.1, 1])
assert s.best_params_["logisticregression__C"] in [0.1, 1] and s.best_score_ > 0.9
