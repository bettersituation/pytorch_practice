import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, inputs):
        h_relu = self.input_linear(x).clamp(min=0)
        for i in range(random.randint(0, 10)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
            print('!', end='')
        print()
        y_pred = self.output_linear(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

criteria = torch.nn.MSELoss(reduction='sum')
opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    y_pred = model(x)

    loss = criteria(y_pred, y)
    print(t, loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
