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


class ConvModelButTraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # Replaced MaxPool2d with AdaptiveAvgPool2d
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Correctly flattens to (batch_size, 64)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplicatedBranchyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: [B, 3, 32, 32]

        self.stem_conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.stem_relu = nn.ReLU()

        # Shared conv reused twice on two different branches.
        self.shared_conv = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.branch1_conv = nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.branch2_conv = nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.branch_relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After pooling:
        # branch1: [B, 12, 16, 16]
        # branch2: [B, 12, 16, 16]
        # skip:    [B, 8, 16, 16]
        #
        # concat:  [B, 32, 16, 16]
        self.merge_conv = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.post_relu = nn.ReLU()

        # Reused Linear module later.
        self.shared_fc = nn.Linear(16 * 16 * 16, 64)

        self.head_a = nn.Linear(64, 10)
        self.head_b = nn.Linear(64, 10)

        self.final_relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 3, 32, 32]

        x = self.stem_conv(x)          # [B, 8, 32, 32]
        x = self.stem_relu(x)          # [B, 8, 32, 32]

        # Fan-out from stem.
        shared_a = self.shared_conv(x) # [B, 8, 32, 32]
        shared_b = self.shared_conv(x) # [B, 8, 32, 32]
        shared_b = self.branch_relu(shared_b)

        # Residual-style add from reused conv calls.
        shared = shared_a + shared_b   # [B, 8, 32, 32]

        # Branch 1: 3x3 conv -> relu -> pool.
        b1 = self.branch1_conv(shared) # [B, 12, 32, 32]
        b1 = self.branch_relu(b1)      # [B, 12, 32, 32]
        b1 = self.pool(b1)             # [B, 12, 16, 16]

        # Branch 2: 1x1 conv -> relu -> avg pool function.
        b2 = self.branch2_conv(shared) # [B, 12, 32, 32]
        b2 = F.relu(b2)                # [B, 12, 32, 32]
        b2 = F.avg_pool2d(b2, kernel_size=2, stride=2)  # [B, 12, 16, 16]

        # Skip branch: pool the shared activation.
        skip = self.pool(shared)       # [B, 8, 16, 16]

        # Multi-input function node.
        merged = torch.cat([b1, b2, skip], dim=1)  # [B, 32, 16, 16]

        merged = self.merge_conv(merged)           # [B, 16, 16, 16]
        merged = self.post_relu(merged)            # [B, 16, 16, 16]

        # Method nodes for testing attrs.
        # NHWC transform then back to NCHW.
        nhwc = merged.permute(0, 2, 3, 1)          # [B, 16, 16, 16]
        nchw = nhwc.permute(0, 3, 1, 2)            # [B, 16, 16, 16]

        flat = nchw.reshape(nchw.shape[0], -1)     # [B, 4096]

        # Reused Linear instance.
        z1 = self.shared_fc(flat)                  # [B, 64]
        z2 = self.shared_fc(flat)                  # [B, 64]

        z = z1 + z2                                # [B, 64]
        z = self.final_relu(z)                     # [B, 64]

        # Two heads.
        out_a = self.head_a(z)                     # [B, 10]
        out_b = self.head_b(z)                     # [B, 10]

        # Final concat for another multi-input op.
        out = torch.cat([out_a, out_b], dim=1)     # [B, 20]

        return out

model = ComplicatedBranchyCNN()
summary = model_summary(model, torch.randn(2, 3, 32, 32))
# Save output to frontend/public/branchy.json
with open("frontend/public/branchy.json", "w") as f:
    json.dump(summary, f)