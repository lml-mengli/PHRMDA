# -*- coding: utf-8 -*-
import torch.nn.functional as F
# from dgl.ops import edge_softmax
from utils_ import *
from torch_geometric.nn import GATConv, SAGEConv, to_hetero, HANConv, HeteroConv, GCNConv, GATv2Conv, GraphConv,RGCNConv
from torch import Tensor, nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperPropagate(nn.Module):
    def __init__(self, in_dim, hyper_dim, act_func='leakyRelu', dropout=None):
        super(HyperPropagate, self).__init__()

        self.act_func = act_func
        self.dropout = dropout

        # 三层FC（对应你lat2, lat3, lat4）
        self.fc1 = nn.Linear(in_dim, hyper_dim, bias=False)
        self.fc2 = nn.Linear(hyper_dim, hyper_dim, bias=False)
        self.fc3 = nn.Linear(hyper_dim, hyper_dim, bias=False)

        # leaky参数（对应你全局leaky）
        self.leaky = 0.1

    # ===== 激活函数 =====
    def activate(self, x):
        if self.act_func == 'relu':
            return F.relu(x)
        elif self.act_func == 'sigmoid':
            return torch.sigmoid(x)
        elif self.act_func == 'tanh':
            return torch.tanh(x)
        elif self.act_func == 'softmax':
            return F.softmax(x, dim=-1)
        elif self.act_func == 'leakyRelu':
            return F.leaky_relu(x, negative_slope=self.leaky)
        elif self.act_func == 'relu6':
            return torch.clamp(x, 0.0, 6.0)
        elif self.act_func == 'relu3':
            return torch.clamp(x, 0.0, 3.0)
        else:
            raise ValueError("Unknown activation")

    # ===== FC封装（替代你原来的FC函数）=====
    def fc_layer(self, x, fc):
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = fc(x)
        x = self.activate(x)
        return x

    # ===== 超图传播 =====
    def forward(self, lats, adj):
        """
        lats: (N, d)
        adj: (N, M)  # 超图关联矩阵 H
        """

        # z0 = H^T * E
        lat1 = self.activate(adj.t() @ lats)  # (M, d)

        # === 三层残差FC ===
        lat2 = self.fc_layer(lat1, self.fc1) + lat1
        lat3 = self.fc_layer(lat2, self.fc2) + lat2
        lat4 = self.fc_layer(lat3, self.fc3) + lat3

        # H * Z
        ret = self.activate(adj @ lat4)  # (N, d)

        return ret



class StructureAwareAdaptiveDiffusion(nn.Module):
    def __init__(self,in_features, L=4, alpha=0.5, topk=10, eps=1e-8):
        super().__init__()
        self.L = L
        # self.alpha = alpha
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.topk = topk
        self.eps = eps
        # learnable hop attention
        # self.hop_weight = nn.Parameter(torch.ones(L))
        self.hop_predictor = nn.Sequential(
            nn.Linear(in_features+1, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, L)
        )
    def forward(self, adj, x):

        # 1. feature similarity
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm,  x_norm.t() )
        sim_matrix = F.relu(sim_matrix)
        sim_matrix = sim_matrix.clone()
        sim_matrix.fill_diagonal_(0)

        # 2. top-k sparsification
        topk_values, topk_indices = torch.topk(sim_matrix,  k=self.topk, dim=1)
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(1, topk_indices, 1)
        sim_matrix = sim_matrix * mask
        sim_matrix = (sim_matrix + sim_matrix.t()) / 2

        # 3. fusion graph
        alpha = torch.sigmoid(self.alpha)
        A_fusion = alpha * adj + (1 - alpha) * sim_matrix

        A_fusion = A_fusion.clone()
        A_fusion.fill_diagonal_(1)

        # 4. random walk matrix
        degree = A_fusion.sum(dim=1, keepdim=True)
        # Row-normalized transition matrix
        P = A_fusion / (degree + self.eps)
        # D_inv = torch.diag(1.0 / (degree + self.eps))
        # # D_inv = torch.diag(torch.pow(degree + self.eps, -0.5))
        # P = torch.mm(D_inv, A_fusion)

        # Step5 Structure-aware Node-adaptive Hop Prediction

        hop_input = torch.cat([x, degree], dim=1)
        hop_att = self.hop_predictor(hop_input)
        hop_att = F.softmax(hop_att, dim=1)

        # Step6 Multi-hop Diffusion
        P_list = []
        P_power = P
        for k in range(self.L):
            if k > 0:
                P_power = torch.mm(P_power, P)
            P_list.append(P_power)
        W = torch.zeros_like(adj)
        for k in range(self.L):
            W += hop_att[:, k].unsqueeze(1) * P_list[k]

        # Step7 Normalize
        W = W / (W.sum(dim=1, keepdim=True) + self.eps)

        return W





class PHRGNN(nn.Module):
    def __init__(self, hidden_size, n_in_features, n_drug,num_heads, dropout, args, data):
        super(PHRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.n_in_features = n_in_features
        self.n_drug = n_drug
        self.hid_features = 32  # 256
        self.hid_out = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.threshold = args.threshold

        self.hetero_transformerEncoder = Attention(d_model=self.n_in_features, n_heads=2, n_layers=1)

        self.hyperPropagate = HyperPropagate(hidden_size, hidden_size)

        self.fc_m = nn.Linear(data["micro"].num_nodes, hidden_size)
        self.fc_d = nn.Linear(data["drug"].num_nodes, hidden_size)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)

        self.alpha = args.alpha

        self.gnn_layer = args.gnn_layer

        self.mic_Embed = nn.Embedding(data["micro"].num_nodes, self.hidden_size)
        self.dru_Embed = nn.Embedding(data["drug"].num_nodes, self.hidden_size)


        self.mhyper = nn.Parameter(torch.randn(self.hidden_size, args.hyperNum))
        self.dhyper = nn.Parameter(torch.randn(self.hidden_size, args.hyperNum))

        self.compute_inf = StructureAwareAdaptiveDiffusion(self.n_in_features,L=4)
        self.fusion = nn.Sequential( nn.Linear(self.hid_out * 2, self.hid_out), nn.ReLU(), nn.Linear(self.hid_out, 1))


        self.gcn1 = GCNConv(in_channels=self.n_in_features, out_channels=self.hid_out)  # 256，64

        self.gcn3 = GCNConv(in_channels=self.n_in_features, out_channels=self.hid_out)  # 256，64

        self.gcn5 = GCNConv(in_channels=self.n_in_features, out_channels=self.hid_out)  # 256，64

        self.relu = nn.ReLU()


    def decode(self,z, edge_label_index):

        return (z['m'][edge_label_index[0]] * z['d'][edge_label_index[1]]).sum(dim=-1)

    def init_weight(self,args):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()

        layers = [args.hidden_size] + [int(self.hid_out/2),int(self.hid_out/2)]

        for k in range(2):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        # return embedding_dict, weight_dict
        return weight_dict


    def forward(self, train_data,x,edge_idx,A_hat,edge_label_index,hyper_perm=False):


        overall_m_embed = self.fc_m(train_data["micro"].x)
        overall_d_embed = self.fc_d(train_data["drug"].x)

        mmHyper = overall_m_embed @ self.mhyper
        ddHyper = overall_d_embed @ self.dhyper


        mmHyper_dropped = F.dropout(mmHyper, p=0.1, training=self.training)
        ddHyper_dropped = F.dropout(ddHyper, p=0.1, training=self.training)

        global_m = self.hyperPropagate(overall_m_embed, mmHyper_dropped)
        global_d = self.hyperPropagate(overall_d_embed, ddHyper_dropped)

        nodes_embeddings={}
        embd_tmp = x

        embd_hetro = self.hetero_transformerEncoder(embd_tmp)


        adj_inf = self.compute_inf(embd_tmp,embd_hetro)
        hete_ass_idx = torch.nonzero(adj_inf)
        hete_ass_weight = adj_inf[hete_ass_idx[:, 0], hete_ass_idx[:, 1]]
        emb1 = self.relu(self.gcn1(embd_hetro, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)


        hete_ass_idx = torch.nonzero(embd_tmp > self.threshold).to(torch.long)
        hete_ass_weight = embd_tmp[hete_ass_idx[:, 0], hete_ass_idx[:, 1]]
        emb5 = self.relu(self.gcn5(embd_hetro, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)

        gate = torch.sigmoid(self.fusion(torch.cat([emb1, emb5], dim=1)))
        emb_all = gate * emb1 + (1 - gate) * emb5
        drug_z = emb_all[0:self.n_drug, :]
        mic_z = emb_all[self.n_drug:, :]


        global_m = F.normalize(global_m, dim=1)
        global_d = F.normalize(global_d, dim=1)

        features_m = torch.stack([global_m, mic_z], dim=1)
        features_d = torch.stack([global_d, drug_z], dim=1)
        mLoss = AMCLoss_v2(features_m,2)
        dLoss = AMCLoss_v2(features_d,2)
        sslloss = mLoss + dLoss
        drug_features1 = torch.cat([global_d,drug_z], dim=1)
        mic_features1 = torch.cat([global_m, mic_z], dim=1)
        nodes_embeddings['d'] = drug_features1
        nodes_embeddings['m'] = mic_features1
        return nodes_embeddings, sslloss



