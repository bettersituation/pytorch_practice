import numpy as np
import torch

if __name__ == '__main__':
    x = torch.empty(5, 3)
    print(x)

    x = torch.rand(5, 3)
    print(x)

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = torch.tensor([5.5, 3])
    print(x)

    x = x.new_ones(5, 3, dtype=torch.double)
    print(x)

    x = torch.randn_like(x, dtype=torch.float)
    print(x)

    print(x.size())

    y = torch.rand(5, 3)
    print(x + y)

    x_tmp = torch.ones(5, 3, dtype=torch.short)
    print(x_tmp)
    try:
        print(x_tmp + y)
    except RuntimeError:
        print('not support dynamic casting operation')

    z = torch.empty(5, 3, dtype=torch.float)
    torch.add(x, y, out=z)
    print(z)

    z.add_(torch.ones(5, 3, dtype=torch.float))
    print(z)

    z_flat = z.view(-1, 1)
    print(z_flat.size())

    np_z = z.numpy()
    print(np_z)

    z.add_(100)
    print(np_z)

    a = np.random.rand(2, 4)
    print(a)

    b = torch.from_numpy(a)
    print(b)

    np.add(a, -100, out=a)
    print(b)

    print(torch.cuda.is_available())

    device = torch.device('cuda')

    p_one = torch.ones_like(b, device=device)
    p = b.to(device)

    np.add(a, 1000, out=a)
    print(b)
    print(p)

    q = b.to(device)

    try:
        print(b + q)
    except RuntimeError:
        print('cpu obj and cuda obj explicitely identifiable and not support operation between them')

    r = p + q
    print(r)

    r = r.to('cpu', dtype=torch.double)
    print(r + b)
