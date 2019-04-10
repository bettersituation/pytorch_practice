import numpy as np
import torch

x = torch.ones(size=(2, 2), requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y ** 2
print(z)

out = z.mean()
print(out)

x.requires_grad_(False)
print(x.grad_fn)

x.requires_grad_(True)
print(x.grad_fn)

out.backward()
print(x.grad)

a = torch.randn(3, dtype=torch.double, requires_grad=True)
b = 2 * a
while b.data.norm() < 1:
    b *= 2
print(a)
print(b)
print(b.data.norm())

v = torch.tensor([0.1, 2, -1], dtype=torch.double)
b.backward(v)
print(a.grad)

print(b.requires_grad)
with torch.no_grad():
    print(b.requires_grad)
    c = b * 1
    print(c.requires_grad)
print(c.requires_grad)
