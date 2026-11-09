# Exercise 1
import torch

q = torch.tensor([[[1.0, 1.0]]])
torch.testing.assert_close(attention_logits(q, q), torch.tensor([[[2**0.5]]]))

# Exercise 2
import torch


s = torch.ones(2, 2, requires_grad=True)
p = masked_softmax(s, torch.tensor([[True, False], [False, False]]))
assert p.tolist() == [[1, 0], [0, 0]]
p.sum().backward()
assert torch.isfinite(s.grad).all()

# Exercise 3
assert causal_mask(2, 1).tolist() == [[True, True, False], [True, True, True]]

# Exercise 4
import torch

x = torch.arange(12).reshape(1, 3, 4)
assert split_heads(x, 2)[0, 1].tolist() == [[2, 3], [6, 7], [10, 11]]
with raises(ValueError):
    split_heads(x, 3)

# Exercise 5
import torch

x = torch.arange(12).reshape(1, 3, 2, 2)
assert torch.equal(join_heads(x.transpose(1, 2)), x.reshape(1, 3, 4))

# Exercise 6
import torch

o, w = attend(torch.zeros(1, 1, 2), torch.ones(1, 2, 2), torch.tensor([[[2.0], [4.0]]]))
assert o.item() == 3 and w.tolist() == [[[0.5, 0.5]]]

# Exercise 7
import torch


m = CausalAttention(8, 2)
x = torch.randn(2, 4, 8)
z = x.clone()
z[:, -1] += 10
torch.testing.assert_close(m(x)[:, :-1], m(z)[:, :-1])
assert m(x).shape == x.shape

# Exercise 8
import torch


x = torch.ones(1, 2, 1, 3)
k, v = append_kv(append_kv(None, x, x), 2 * x, 3 * x)
assert k.shape == (1, 2, 2, 3) and k[0, 0, -1, 0] == 2 and v[0, 0, -1, 0] == 3

# Exercise 9
import math


p = positions(3, 5)
assert p.shape == (3, 5) and p[0].tolist() == [0, 1, 0, 1, 0]
assert abs(p[1, 0].item() - math.sin(1)) < 1e-6

# Exercise 10
from torch import nn

import torch


m = Block(6, nn.Linear(6, 6))
x = torch.randn(2, 3, 6, requires_grad=True)
m(x).sum().backward()
assert x.grad.shape == x.shape and all(p.grad is not None for p in m.parameters())
