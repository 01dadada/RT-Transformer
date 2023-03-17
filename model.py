import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv
from torch.nn.functional import sigmoid
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from torch.nn import Conv1d


class GraphTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, edge_dim=5, dropout=0, **kwargs):
        super(GraphTransformerBlock, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = GATConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim)
        self.linear = nn.Linear(heads * out_channels, out_channels)
        self.layerNorm = nn.LayerNorm(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):

        x_gat = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x_gat = self.linear(x_gat)
        x_gat = self.layerNorm(x + x_gat)

        return F.dropout(x_gat, self.dropout, training=self.training)


class GraphTransformerBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, edge_dim=5, dropout=0, **kwargs):
        super(GraphTransformerBlock2, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = GATConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim)
        self.linear1 = nn.Linear(heads * out_channels, out_channels)
        self.layerNorm1 = nn.LayerNorm(out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.layerNorm2 = nn.LayerNorm(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x_gat = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x_gat = self.linear1(x_gat)
        x_gat = self.layerNorm1(x + x_gat)
        linear_ = self.linear2(x_gat)
        linear_ = self.layerNorm2(linear_ + x_gat)

        return F.dropout(linear_, self.dropout, training=self.training)

class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10,
                                                   verbose=False, threshold=0.0001, threshold_mode='rel',
                                                   cooldown=0, min_lr=0, eps=1e-08)

        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for i, data in enumerate(data_loader):
            data.to(self.device)
            y_hat = self.model(data)
            loss = criterion(y_hat, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return 0


class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    # def test_regressor(self, data_loader):
    #     total_loss = torch.tensor(0, dtype=torch.float64, device=self.device)
    #     with torch.no_grad():
    #         for data in tqdm(data_loader):
    #             data.to(self.device)
    #             y_hat = self.model(data)
    #             total_loss += torch.abs(y_hat - data.y).sum()
    #     return total_loss
    def test_regressor(self, data_loader):
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device, non_blocking=True)
                y_hat = self.model(data)
                # total_loss += torch.abs(y_hat - data.y).sum()
                # mre_total = torch.div(torch.abs(y_hat - data.y), data.y).sum()
                y_true.append(data.y)
                y_pred.append(y_hat)

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            mae = torch.abs(y_true - y_pred).sum()
            # mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            # medAE = torch.median(torch.abs(y_true - y_pred))
            # medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
            #
            # score = torchmetrics.R2Score().to(self.device)
            # r2 = score(y_pred, y_true)
        # return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()
        return mae


class MyNet(nn.Module):
    def __init__(self, emb_dim=512, feat_dim=256, edge_dim=5, heads=3, drop_ratio=0, pool='add'):
        super(MyNet, self).__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.in_linear = nn.Linear(34, emb_dim)

        self.conv1 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv3 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv4 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv5 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv6 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv7 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv8 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv9 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 8, self.feat_dim // 64),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 64, 1),
        )

        self.conv1d1 = OneDimConvBlock()
        self.conv1d2 = OneDimConvBlock()
        self.conv1d3 = OneDimConvBlock()
        self.conv1d4 = OneDimConvBlock()
        self.conv1d5 = OneDimConvBlock()
        self.conv1d6 = OneDimConvBlock()
        self.conv1d7 = OneDimConvBlock()
        self.conv1d8 = OneDimConvBlock()
        self.conv1d9 = OneDimConvBlock()
        self.conv1d10 = OneDimConvBlock()
        self.conv1d11 = OneDimConvBlock()
        self.conv1d12 = OneDimConvBlock()

        self.preconcat1 = nn.Linear(2048, 1024)
        self.preconcat2 = nn.Linear(1024, self.feat_dim)

        self.afterconcat1 = nn.Linear(2 * self.feat_dim, self.feat_dim)
        self.after_cat_drop = nn.Dropout(self.drop_ratio)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        fringerprint = data.fingerprint.reshape(-1, 2048)

        h = self.in_linear(x)

        h = F.relu(self.conv1(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv2(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv3(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv4(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv5(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv6(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv7(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv8(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv9(h, edge_index, edge_attr), inplace=True)

        fringerprint = self.conv1d1(fringerprint)
        fringerprint = self.conv1d2(fringerprint)
        fringerprint = self.conv1d3(fringerprint)
        fringerprint = self.conv1d4(fringerprint)
        fringerprint = self.conv1d5(fringerprint)
        fringerprint = self.conv1d6(fringerprint)
        fringerprint = self.conv1d7(fringerprint)
        fringerprint = self.conv1d8(fringerprint)
        fringerprint = self.conv1d9(fringerprint)
        fringerprint = self.conv1d10(fringerprint)
        fringerprint = self.conv1d11(fringerprint)
        fringerprint = self.conv1d12(fringerprint)
        fringerprint = self.preconcat1(fringerprint)
        fringerprint = self.preconcat2(fringerprint)

        h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        h = self.pool(h, batch)
        h = self.feat_lin(h)

        concat = torch.concat([h, fringerprint], dim=-1)
        concat = self.afterconcat1(concat)
        concat = self.after_cat_drop(concat)

        out = self.out_lin(concat)

        return out.squeeze()


class OneDimConvBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=2048):
        super().__init__()
        self.attention_conv = OneDimAttention(in_channel, in_channel)
        self.batchnorm1 = torch.nn.BatchNorm1d(in_channel)
        self.batchnorm2 = torch.nn.BatchNorm1d(in_channel)
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.linear2 = nn.Linear(in_channel, out_channel)
        self.ffn = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.attention_conv(x, x, x)
        h = self.batchnorm1(x + h)

        h_new = self.ffn(h)
        h_new = self.batchnorm2(h + h_new)
        return F.dropout1d(self.linear2(h_new), training=self.training)


class OneDimAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = torch.tensor(in_size)
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, q, k, v):
        attention = torch.mul(q, k) / torch.sqrt(self.in_size)
        attention = self.linear(attention)
        return torch.mul(F.softmax(attention, dim=-1), v)


class MyNetTest(nn.Module):
    def __init__(self, emb_dim=512, feat_dim=256, edge_dim=5, heads=3, drop_ratio=0, pool='add'):
        super(MyNetTest, self).__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.in_linear = nn.Linear(34, emb_dim)

        self.conv1 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv3 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv4 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv5 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv6 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv7 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv8 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv9 = GraphTransformerBlock2(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 8, self.feat_dim // 64),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 64, 1),
        )

        self.conv1d1 = OneDimConvBlock()
        self.conv1d2 = OneDimConvBlock()
        self.conv1d3 = OneDimConvBlock()
        self.conv1d4 = OneDimConvBlock()
        self.conv1d5 = OneDimConvBlock()
        self.conv1d6 = OneDimConvBlock()
        self.conv1d7 = OneDimConvBlock()
        self.conv1d8 = OneDimConvBlock()
        self.conv1d9 = OneDimConvBlock()
        self.conv1d10 = OneDimConvBlock()
        self.conv1d11 = OneDimConvBlock()
        self.conv1d12 = OneDimConvBlock()

        self.preconcat1 = nn.Linear(2048, 1024)
        self.preconcat2 = nn.Linear(1024, self.feat_dim)

        self.afterconcat1 = nn.Linear(2 * self.feat_dim, self.feat_dim)
        self.after_cat_drop = nn.Dropout(self.drop_ratio)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        fringerprint = data.fingerprint.reshape(-1, 2048)

        h = self.in_linear(x)

        h = F.relu(self.conv1(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv2(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv3(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv4(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv5(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv6(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv7(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv8(h, edge_index, edge_attr), inplace=True)
        h = F.relu(self.conv9(h, edge_index, edge_attr), inplace=True)

        fringerprint = self.conv1d1(fringerprint)
        fringerprint = self.conv1d2(fringerprint)
        fringerprint = self.conv1d3(fringerprint)
        fringerprint = self.conv1d4(fringerprint)
        fringerprint = self.conv1d5(fringerprint)
        fringerprint = self.conv1d6(fringerprint)
        fringerprint = self.conv1d7(fringerprint)
        fringerprint = self.conv1d8(fringerprint)
        fringerprint = self.conv1d9(fringerprint)
        fringerprint = self.conv1d10(fringerprint)
        fringerprint = self.conv1d11(fringerprint)
        fringerprint = self.conv1d12(fringerprint)
        fringerprint = self.preconcat1(fringerprint)
        fringerprint = self.preconcat2(fringerprint)

        h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        h = self.pool(h, batch)
        h = self.feat_lin(h)

        concat = torch.concat([h, fringerprint], dim=-1)
        concat = self.afterconcat1(concat)
        concat = self.after_cat_drop(concat)

        out = self.out_lin(concat)

        return out.squeeze()