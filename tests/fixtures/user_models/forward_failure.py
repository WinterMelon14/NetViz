import torch


class UserModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("fixture forward failure")

