import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def known_model_input():
    return torch.randn(1, 4)
