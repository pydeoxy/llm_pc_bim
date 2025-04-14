import torch
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

import loralib as lora

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

# Point Cloud data with colors in the x features (r,g,b,nx,ny,nz)
class PyGPointNet2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 6, 64, 64, 128])) # 3 (pos) + 6 (x)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 6, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)

# Point Cloud data without colors in the x features (nx,ny,nz)
class PyGPointNet2NoColor(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128])) # 3 (pos) + 3 (x)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128])) #

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)

# Model for LoRa training, lora weights added
class PointNet2LoRa(torch.nn.Module):
    def __init__(self, num_classes, lora_r=8, lora_alpha=16):
        super().__init__()

        # Replace linear layers in SAModule/FPModule MLPs with LoRA
        self.sa1_module = SAModule(0.2, 0.2, self._make_lora_mlp([3 + 6, 64, 64, 128], lora_r, lora_alpha))
        self.sa2_module = SAModule(0.25, 0.4, self._make_lora_mlp([128 + 3, 128, 128, 256], lora_r, lora_alpha))
        self.sa3_module = GlobalSAModule(self._make_lora_mlp([256 + 3, 256, 512, 1024], lora_r, lora_alpha))

        self.fp3_module = FPModule(1, self._make_lora_mlp([1024 + 256, 256, 256], lora_r, lora_alpha))
        self.fp2_module = FPModule(3, self._make_lora_mlp([256 + 128, 256, 128], lora_r, lora_alpha))
        self.fp1_module = FPModule(3, self._make_lora_mlp([128 + 6, 128, 128, 128], lora_r, lora_alpha))

        # Replace final MLP and linear layers with LoRA
        self.mlp = self._make_lora_mlp([128, 128, 128, num_classes], lora_r, lora_alpha, dropout=0.5)
        self.lin1 = lora.Linear(128, 128, r=lora_r, lora_alpha=lora_alpha)
        self.lin2 = lora.Linear(128, 128, r=lora_r, lora_alpha=lora_alpha)
        self.lin3 = lora.Linear(128, num_classes, r=lora_r, lora_alpha=lora_alpha)

    def _make_lora_mlp(self, channels, r, alpha, dropout=0.0):
        # Helper to create MLP with LoRA layers
        layers = []
        for i in range(len(channels) - 1):
            layers.append(lora.Linear(channels[i], channels[i+1], r=r, lora_alpha=alpha))
            if i < len(channels) - 2:
                layers.append(torch.nn.ReLU())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, data):
        # Unchanged forward pass
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)