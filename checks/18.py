# Exercise 1
a = inference_key("A", "m1", "t1", "hi", {})
assert (
    len(a) == 64
    and a != inference_key("B", "m1", "t1", "hi", {})
    and a != inference_key("A", "m1", "t1", "hi", {"seed": 1})
)

# Exercise 2
c = ResultCache(2)
assert c.get_or_compute("a", lambda: None) is None
c.get_or_compute("b", lambda: 2)
c.get_or_compute("a", None)
c.get_or_compute("c", lambda: 3)
assert c.get_or_compute("b", lambda: 9) == 9

# Exercise 3
import numpy as np

q, s = quantize_rows([[2, -2], [0, 0]])
assert q.dtype == np.int8 and q.tolist() == [[127, -127], [0, 0]]
np.testing.assert_allclose(q * s, [[2, -2], [0, 0]])

# Exercise 4
assert kv_bytes(2, 3, 4, 1, 8, 2) == 768
with raises(ValueError):
    kv_bytes(1, 1, -1, 1, 1, 1)

# Exercise 5
import torch

w = torch.zeros(2, 2)
out = merge_lora(w, torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0], [2.0]]), 2)
assert out.tolist() == [[2, 4], [4, 8]] and w.sum() == 0

# Exercise 6
cfg = lora_config(["q_proj", "v_proj"], 4, 8)
assert (
    cfg.r == 4
    and cfg.target_modules == {"q_proj", "v_proj"}
    and cfg.task_type == "CAUSAL_LM"
)

# Exercise 7
cmd = vllm_command("/models/test", 2048, 2, 0.8)
assert (
    cmd[:3] == ["vllm", "serve", "/models/test"]
    and cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
)
with raises(ValueError):
    vllm_command("m", 2048, 1, 1.4)

# Exercise 8
import sys, torch
from unittest.mock import Mock, patch

model, batch, compiled = Mock(), Mock(), Mock()
with (
    patch.dict(sys.modules, {"torch_tensorrt": Mock()}),
    patch.object(torch.cuda, "is_available", return_value=True),
    patch.object(torch, "compile", return_value=compiled) as compile_model,
):
    assert tensorrt_model(model, batch) is compiled
    compile_model.assert_called_once_with(
        model.eval.return_value.to.return_value, backend="tensorrt"
    )
    compiled.assert_called_once_with(batch.to.return_value)

# Exercise 9
assert token_batches([2, 4, 3, 6], 8) == [[2, 4], [3], [6]]
with raises(ValueError):
    token_batches([9], 8)

# Exercise 10
ticks = iter([0, 1, 2, 5])
calls = []
r = latency(lambda: calls.append(1), lambda: None, lambda: next(ticks), 1, 2)
assert r["median"] == 2 and abs(r["p95"] - 2.9) < 1e-9 and len(calls) == 3
