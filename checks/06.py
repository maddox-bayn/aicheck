# Exercise 1
import torch

m = mlp(4, 8, 3).eval()
x = torch.ones(2, 4)
assert m(x).shape == (2, 3) and torch.equal(m(x), m(x))

# Exercise 2
from torch import nn

import torch


m = Residual(3)
for p in m.parameters():
    nn.init.zeros_(p)
x = torch.randn(4, 3)
assert torch.equal(m(x), x)

# Exercise 3
import torch

y = conv_block(3, 6)(torch.randn(2, 3, 9, 7))
assert y.shape == (2, 6, 9, 7) and y.min() >= 0

# Exercise 4
import torch

m = image_net(3, 4)
assert all(m(torch.ones(2, 3, h, w)).shape == (2, 4) for h, w in [(12, 9), (7, 6)])

# Exercise 5
import torch

m = separable(4, 7)
assert m(torch.ones(2, 4, 8, 8)).shape == (2, 7, 8, 8)
assert sum(p.numel() for p in m.parameters()) == 4 * 9 + 4 * 7

# Exercise 6
from torch import nn

m = freeze(nn.Sequential(nn.Linear(2, 2), nn.Dropout()))
assert (
    not m.training
    and not m[1].training
    and all(not p.requires_grad for p in m.parameters())
)

# Exercise 7
import torch


m = balanced_loss([1, 4])
assert torch.isclose(m.weight[0] / m.weight[1], torch.tensor(4.0))
with raises(ValueError):
    balanced_loss([1, 0])

# Exercise 8
import torch


from torch import nn

m = nn.Identity()
assert (
    accuracy(
        m,
        [
            (torch.eye(2), torch.tensor([0, 1])),
            (torch.tensor([[1.0, 0.0]]), torch.tensor([1])),
        ],
    )
    == 2 / 3
    and m.training
)
with raises(ValueError):
    accuracy(m, [])

# Exercise 9
import torch
from torch import nn

m = nn.Linear(2, 2)
saved = best_state(m)
nn.init.constant_(m.weight, 12)
assert not torch.equal(saved["weight"], m.weight)
m.load_state_dict(saved)
assert torch.equal(saved["weight"], m.weight)

# Exercise 10
from torch import nn

import torch


m = nn.Linear(2, 2)
before = m.weight.detach().clone()
loss = train_epoch(
    m, [(torch.eye(2), torch.tensor([0, 1]))], torch.optim.SGD(m.parameters(), lr=0.1)
)
assert loss > 0 and not torch.equal(before, m.weight)
