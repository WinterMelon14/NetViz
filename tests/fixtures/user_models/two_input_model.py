import torch


class UserModel(torch.nn.Module):
    def forward(self, left, right):
        return torch.matmul(left, right)
