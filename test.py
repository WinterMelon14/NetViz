import json
import operator
from typing import Any

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from util import *



class WeirdButTraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.m1 = nn.Linear(3, 2)
        self.m2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x1 = self.m1(x)
        x2 = self.m2(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        return x



class Branchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.module = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x1 = self.module(x)
        x2 = self.module(x)
        x = torch.cat([x1, x2], dim=1)

        return self.relu(x)


model = Branchy()
summary = model_summary(model, torch.randn(1, 4))
print(json.dumps(summary, indent=2))