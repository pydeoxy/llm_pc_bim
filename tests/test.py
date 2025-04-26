import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate

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

layer = SAModule(0.2, 0.2, MLP([3 + 6, 64, 64, 128]))




print(layer.conv.local_nn)
print(hasattr(layer.conv.local_nn, 'lins'))
print(layer.conv.local_nn.lins)
for lin in layer.conv.local_nn.lins:
    print(lin)