import torch


class UserModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.ReLU() for _ in range(1_200))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

