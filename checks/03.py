# Exercise 1
import torch


x = torch.tensor([2.0], requires_grad=True)
y = batch([x, x], "cpu")
y.sum().backward()
assert y.shape == (2, 1) and x.grad.item() == 2

# Exercise 2
import torch

assert class_scores(
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1, 0])
).tolist() == [2, 3]

# Exercise 3
import torch


x = torch.tensor([2.0, 3.0], requires_grad=True)
assert (
    vjp(lambda a: a * a, x, torch.tensor([2.0, 1.0])).tolist() == [8, 6]
    and x.grad is None
)

# Exercise 4
import torch


x = torch.tensor([2.0, 3.0], requires_grad=True)
assert hvp(lambda a: (a**3).sum(), x, torch.ones_like(x)).tolist() == [12, 18]

# Exercise 5
import torch

x = torch.tensor([2.0], requires_grad=True)
y = snapshot(x)
with torch.no_grad():
    x.add_(1)
assert not y.requires_grad and y.item() == 2 and y.device.type == "cpu"

# Exercise 6
import torch


d = Windows(torch.arange(5), 2)
assert len(d) == 3 and d[2][0].tolist() == [2, 3] and d[2][1] == 4
with raises(IndexError):
    d[3]

# Exercise 7
import torch


x, m = collate([torch.tensor([0, 2]), torch.tensor([3])])
assert x.tolist() == [[0, 2], [3, 0]] and m.tolist() == [[True, True], [True, False]]

# Exercise 8
import torch


p = torch.nn.Parameter(torch.ones(2))
p.grad = torch.tensor([3.0, 4.0])
assert abs(clip([p], 1).item() - 5) < 1e-5 and abs(p.grad.norm().item() - 1) < 1e-5

# Exercise 9
import torch

m = torch.nn.Linear(1, 1, bias=False)
with torch.no_grad():
    m.weight.zero_()
x = torch.ones(3, 1)
y = torch.tensor([[1.0], [1.0], [4.0]])
microstep(
    m,
    torch.optim.SGD(m.parameters(), lr=0.1),
    [(x[:2], y[:2]), (x[2:], y[2:])],
    torch.nn.functional.mse_loss,
)
assert abs(m.weight.item() - 0.4) < 1e-5

# Exercise 10
import torch


x = torch.tensor([-1.0, 2.0], dtype=torch.double, requires_grad=True)
assert torch.autograd.gradcheck(Cube.apply, (x,))
