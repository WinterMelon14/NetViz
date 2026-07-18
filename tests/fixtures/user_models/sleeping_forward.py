import time

import torch


class UserModel(torch.nn.Module):
    def forward(self, x):
        time.sleep(60)
        return x

