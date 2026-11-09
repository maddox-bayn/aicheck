# Exercise 1
assert Observation(sensor="a", reading=2.0).reading == 2
with raises(ValueError):
    Observation(sensor="a", reading=float("nan"))
with raises(ValueError):
    Observation(sensor="a", reading=2.0, extra=3)

# Exercise 2
from pydantic import BaseModel


class Row(BaseModel):
    value: int


ok, bad = quarantine([{"value": 1}, {"value": "oops"}], Row)
assert ok == [{"value": 1}] and bad[0][0] == 1 and bad[0][1][0]["loc"] == ("value",)

# Exercise 3
import pandas as pd


s = utc_times(pd.Series(["2025-01-01T01:00:00+01:00", "bad"], index=[4, 8]))
assert s[4].hour == 0 and pd.isna(s[8]) and str(s.dtype).endswith("UTC]")

# Exercise 4
import pandas as pd

f = pd.DataFrame(
    {"event_id": ["a", "a", "a"], "revision": [2, 1, 2], "value": [1, 2, 3]}
)
assert latest(f).value.tolist() == [3]

# Exercise 5
import pandas as pd

f = pd.DataFrame({"id": [1, 2]})
d = pd.DataFrame({"id": [1], "name": ["x"]})
assert pd.isna(enrich(f, d).name.iloc[1])
with raises(pd.errors.MergeError):
    enrich(f, pd.concat([d, d]))

# Exercise 6
import pandas as pd

f = pd.DataFrame(
    {"device": ["a", "b", "a"], "time": [2, 1, 1], "value": [9, 3, 7]}, index=[0, 0, 0]
)
assert add_lag(f).lag.iloc[0] == 7 and add_lag(f).lag.isna().sum() == 2

# Exercise 7
import pandas as pd

idx = pd.to_datetime(["2025-01-01", "2025-01-01 02:00"], utc=True, format="mixed")
assert hourly_counts(pd.DataFrame(index=idx)).tolist() == [1, 0, 1]

# Exercise 8
import pandas as pd

assert impute(
    pd.DataFrame({"x": [1, 5]}), pd.DataFrame({"x": [None, 100]})
).x.tolist() == [3, 100]
with raises(ValueError):
    impute(pd.DataFrame({"x": [float("nan")]}), pd.DataFrame({"x": [None]}))

# Exercise 9
import pandas as pd

f = pd.DataFrame({"account": ["b", "a"], "amount": [3, 2]})
assert (
    list(chunk_totals([f, f]).items()) == [("a", 4), ("b", 6)]
    and chunk_totals([]) == {}
)

# Exercise 10
import pandas as pd


e = pd.DataFrame({"time": [4, 1, 3], "device": ["a", "a", "b"]})
f = pd.DataFrame({"time": [2, 5], "device": ["a", "b"], "v": [10, 20]})
v = point_in_time(e, f).v
assert v.iloc[0] == 10 and v.iloc[1:].isna().all()
