import json
import torch
import torch.nn as nn
from util.summary import model_summary



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

class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()
        
        # Sub-network 1: Image Processing (CNN)
        self.cnn_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten() # Flattens to (batch_size, 16 * 32 * 32) assuming 64x64 input
        )
        
        # Sub-network 2: Tabular Processing (MLP)
        self.mlp_path = nn.Sequential(
            nn.Linear(in_features=10, out_features=32),
            nn.ReLU()
        )
        
        # Final Classifier: Combined feature size = 16384 (CNN) + 32 (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(16 * 32 * 32) + 32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2) # 2 output classes
        )

    def forward(self, image_input, tabular_input):
        # 1. Process paths independently
        x1 = self.cnn_path(image_input)
        x2 = self.mlp_path(tabular_input)
        
        # 2. Concatenate along the feature dimension (dim=1)
        combined = torch.cat((x1, x2), dim=1)
        
        # 3. Final classification
        output = self.classifier(combined)
        return output

class doubleTrackModel(nn.Module):
    def __init__(self):
        super(doubleTrackModel, self).__init__()
        self.track1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.track2 = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )
        self.final_layer = nn.Linear(20, 5)  # Final output layer

    def forward(self, x):
        out1 = self.track1(x)
        out2 = self.track2(x)
        combined = torch.cat((out1, out2), dim=1)  # Concatenate outputs
        final_output = self.final_layer(combined)
        return final_output


class MultiTrackModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.track1 = nn.Sequential(
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
        )

        self.track2 = nn.Sequential(
            nn.Linear(32, 40),
            nn.ReLU(),
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
        )

        self.track3 = nn.Sequential(
            nn.Linear(32, 24),
            nn.GELU(),
        )

        self.track4 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
        )

        self.merge12 = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.merge34 = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        stem = self.stem(x)

        out1 = self.track1(stem)
        out2 = self.track2(stem)
        out3 = self.track3(stem)
        out4 = self.track4(stem)

        merged12 = self.merge12(torch.cat((out1, out2), dim=1))
        merged34 = self.merge34(torch.cat((out3, out4), dim=1))

        combined = torch.cat((merged12, merged34), dim=1)
        return self.final(combined)


class ShortcutHeavyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(16, 32)

        self.branch1_layer1 = nn.Linear(32, 32)
        self.branch1_layer2 = nn.Linear(32, 32)
        self.branch1_layer3 = nn.Linear(32, 32)

        self.branch2_layer1 = nn.Linear(32, 48)
        self.branch2_layer2 = nn.Linear(48, 32)

        self.branch3_layer1 = nn.Linear(32, 24)
        self.branch3_layer2 = nn.Linear(24, 24)
        self.branch3_layer3 = nn.Linear(24, 32)

        self.merge = nn.Linear(96, 32)

        self.post_merge1 = nn.Linear(32, 32)
        self.post_merge2 = nn.Linear(32, 32)

        self.output = nn.Linear(32, 6)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        root = self.relu(self.input_projection(x))

        # Long branch with a root-to-end shortcut.
        branch1 = self.relu(self.branch1_layer1(root))
        branch1 = self.relu(self.branch1_layer2(branch1))
        branch1 = self.branch1_layer3(branch1)
        branch1 = self.relu(branch1 + root)

        # Medium branch with another shortcut.
        branch2 = self.relu(self.branch2_layer1(root))
        branch2 = self.branch2_layer2(branch2)
        branch2 = self.relu(branch2 + root)

        # Longer nonlinear branch.
        branch3 = self.gelu(self.branch3_layer1(root))
        branch3 = self.gelu(self.branch3_layer2(branch3))
        branch3 = self.branch3_layer3(branch3)

        merged = torch.cat(
            (branch1, branch2, branch3),
            dim=1,
        )
        merged = self.relu(self.merge(merged))

        # Another A -> B -> C -> D plus A -> D pattern.
        residual = merged
        out = self.relu(self.post_merge1(merged))
        out = self.post_merge2(out)
        out = self.relu(out + residual)

        return self.output(out)
    
class ComplicatedTrackModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.track_a = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.track_b1 = nn.Linear(64, 96)
        self.track_b2 = nn.Linear(96, 64)
        self.track_b3 = nn.Linear(64, 32)

        self.track_c = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Linear(48, 32),
        )

        self.track_d1 = nn.Linear(64, 80)
        self.track_d2 = nn.Linear(80, 80)
        self.track_d3 = nn.Linear(80, 48)
        self.track_d4 = nn.Linear(48, 32)

        self.track_e = nn.Linear(64, 32)

        self.merge_ab = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
        )

        self.merge_cd = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
        )

        self.middle_merge = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64),
        )

        self.late_track1 = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
        )

        self.late_track2_layer1 = nn.Linear(64, 64)
        self.late_track2_layer2 = nn.Linear(64, 32)

        self.late_track3 = nn.Sequential(
            nn.Linear(64, 40),
            nn.GELU(),
            nn.Linear(40, 40),
            nn.GELU(),
            nn.Linear(40, 32),
        )

        self.final_merge = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        root = self.stem(x)

        # Initial five-way split.
        a = self.track_a(root)

        b = self.relu(self.track_b1(root))
        b = self.relu(self.track_b2(b))
        b = self.track_b3(b)

        c = self.track_c(root)

        d = self.relu(self.track_d1(root))
        d = self.relu(self.track_d2(d))
        d = self.relu(self.track_d3(d))
        d = self.track_d4(d)

        e = self.track_e(root)

        # First pair of merges.
        ab = self.merge_ab(torch.cat((a, b), dim=1))
        cd = self.merge_cd(torch.cat((c, d), dim=1))

        # Merge those results with the untouched fifth track.
        middle = torch.cat((ab, cd, e), dim=1)
        middle = self.middle_merge(middle)

        # Residual shortcut around the middle transformation.
        middle = middle + root

        # Late three-way split.
        late1 = self.late_track1(middle)

        late2 = self.relu(self.late_track2_layer1(middle))
        late2 = self.late_track2_layer2(late2)

        # Shortcut from the late split root to the end of track 2.
        late2 = late2 + middle[:, :32]

        late3 = self.late_track3(middle)

        final_input = torch.cat(
            (late1, late2, late3),
            dim=1,
        )

        return self.final_merge(final_input)
from transcriber import PianoTranscriber, ModelWithHStackOnly, SDPASelfAttention
model = ComplicatedBranchyCNN(d_model=4)
