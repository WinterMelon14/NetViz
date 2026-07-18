import torch


class UserModel(torch.nn.Module):
    def forward(self, x):
        return x.relu()

