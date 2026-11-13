# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import dgl
import pandas as pd
import argparse
import torch.nn.functional as F


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.eps = torch.tensor(1e-5)

    def forward(self, iput, target, gamma):
        loss_sum = torch.pow((iput - target), 2)
        result = (1 - gamma) * ((target * loss_sum).sum()) + gamma * (((1 - target) * loss_sum).sum())
        return (result + self.eps)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


class Matrix(nn.Module):
    def __init__(self):
        super(Matrix, self).__init__()

    def hits(self, pos_index, scores_index):
        if pos_index[0] in scores_index:
            Hits = 1
        else:
            Hits = 0
        return Hits

    def ndcg(self, pos_index, scores_index, n):
        dcg_sum = 0
        idcg_sum = 0
        for j in range(len(scores_index)):
            if scores_index[j] == pos_index[0]:
                dcg_sum += self.dcg(1, j + 1)
            else:
                dcg_sum += self.dcg(0, j + 1)
        for m in range(n):
            if m == 0:
                idcg_sum += self.dcg(1, m + 1)
            else:
                idcg_sum += self.dcg(0, m + 1)
        return dcg_sum, idcg_sum

    def dcg(self, rel, index):
        dcg = (2 ** rel - 1) / np.log2(index + 1)
        return dcg

    def forward(self, n, num, predict_val, num_pos, index):
        sample_hit, sample_ndcg = [], []
        Hits_sum = 0
        ndcg_sum = 0
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]
        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple][:n]
            Hits = self.hits(pos_index, scores_index)
            dcg_sum, idcg_sum = self.ndcg(pos_index, scores_index, n)
            ndcg_sum += dcg_sum / idcg_sum
            Hits_sum += Hits
            sample_hit.append(Hits)
            sample_ndcg.append(dcg_sum / idcg_sum)
        Hits = Hits_sum / num_pos
        ndcg = ndcg_sum / num_pos
        return Hits, ndcg, sample_hit, sample_ndcg


class MRR(nn.Module):
    def __init__(self):
        super(MRR, self).__init__()

    def forward(self, num, predict_val, num_pos, index):
        sample_mrr = []
        rank_sum = 0
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]
        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple]
            sample_mrr.append(1 / (scores_index.index(pos_index[0]) + 1))
            rank_sum += 1 / (scores_index.index(pos_index[0]) + 1)
        mrr = rank_sum / num_pos
        return mrr, sample_mrr


def construct_hg(pos_data):
    g_m_edges, m_d_edges, g_d_edges = [list() for x in range(3)]
    for i in range(len(pos_data)):
        one_g_m_edge = []
        one_g_m_edge.extend(pos_data[i][0:2].tolist())
        one_m_d_edge = []
        one_m_d_edge.extend(pos_data[i][1:3].tolist())
        one_g_d_edge = []
        one_g_d_edge.extend([pos_data[i][0], pos_data[i][2]])
        if not one_g_m_edge in g_m_edges:
            g_m_edges.append(one_g_m_edge)
        if not one_m_d_edge in m_d_edges:
            m_d_edges.append(one_m_d_edge)
        if not one_g_d_edge in g_d_edges:
            g_d_edges.append(one_g_d_edge)
    g_m_edges = np.array(sorted(g_m_edges, key=(lambda x: x[0])))
    m_d_edges = np.array(sorted(m_d_edges, key=(lambda x: x[0])))
    g_d_edges = np.array(sorted(g_d_edges, key=(lambda x: x[0])))
    hg = dgl.heterograph({
        ('g', 'g_m', 'm'): (torch.LongTensor(g_m_edges[:, 0]), torch.LongTensor(g_m_edges[:, 1])),
        ('m', 'm_d', 'd'): (torch.LongTensor(m_d_edges[:, 0]), torch.LongTensor(m_d_edges[:, 1])),
        ('g', 'g_d', 'd'): (torch.LongTensor(g_d_edges[:, 0]), torch.LongTensor(g_d_edges[:, 1])),
        ('m', 'm_g', 'g'): (torch.LongTensor(g_m_edges[:, 1]), torch.LongTensor(g_m_edges[:, 0])),
        ('d', 'd_m', 'm'): (torch.LongTensor(m_d_edges[:, 1]), torch.LongTensor(m_d_edges[:, 0])),
        ('d', 'd_g', 'g'): (torch.LongTensor(g_d_edges[:, 1]), torch.LongTensor(g_d_edges[:, 0]))
    })
    return hg


class Prevent_leakage(nn.Module):
    def __init__(self, test_data):
        super(Prevent_leakage, self).__init__()
        self.test_data = test_data

    def forward(self, metapath_instances):
        test_pos_data = pd.DataFrame(self.test_data[:, :3], columns=['g', 'm', 'd'])
        metapath_instances_all = metapath_instances.append(test_pos_data)
        metapath_instances_all = metapath_instances_all.append(test_pos_data)
        exclude_metapath_instances = metapath_instances_all.drop_duplicates(subset=['g', 'm', 'd'], keep=False)
        exclude_metapath_instances = exclude_metapath_instances.reset_index(drop=True)
        return exclude_metapath_instances


class Separate_subgraph(nn.Module):
    def __init__(self):
        super(Separate_subgraph, self).__init__()

    def get_edges(self, edges1, edges2):
        new_edges = [[list() for j in range(2)] for i in range(2)]
        for i in range(len(edges1[0])):
            if edges1[1][i] in edges2[0]:
                new_edges[0][0].append(edges1[0][i])
                new_edges[0][1].append(edges1[1][i])
                index = [m for m, x in enumerate(edges2[0]) if x == edges1[1][i]]
                if edges1[1][i] not in new_edges[1][0]:
                    for j in range(len(index)):
                        new_edges[1][0].append(edges1[1][i])
                        new_edges[1][1].append(edges2[1][index[j]])
        return new_edges

    def forward(self, hg, metapath):
        new_triplets_edge = []
        metapath_list = [f"{metapath[i]}_{metapath[i + 1]}" for i in range(len(metapath) - 1)]
        edges = [hg.edges(etype=metapath_list[i]) for i in range(len(metapath_list))]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]

        if len(metapath_list) == 2:
            new_edges = self.get_edges(edges[0], edges[1])
        elif len(metapath_list) == 3:
            new_edges = self.get_edges(edges[0], edges[1])
            new_edges1 = self.get_edges(new_edges[1], edges[2])
            new_edges.append(new_edges1[1])
        for path in metapath_list:
            for i in range(len(hg.canonical_etypes)):
                if path in hg.canonical_etypes[i]:
                    new_triplets_edge.append(hg.canonical_etypes[i])
        graph_data = {}
        for i in range(len(metapath_list)):
            graph_data[new_triplets_edge[i]] = (new_edges[i][0], new_edges[i][1])
        subgraph = dgl.heterograph(graph_data)
        return subgraph


def ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix, epoch_max_matrix, e, hits_1,
              hits_3, hits_5, ndcg1, ndcg3, ndcg5, MRR):
    if hits_1 >= hits_max_matrix[0][0]:
        hits_max_matrix[0][0] = hits_1
        hits_max_matrix[0][1] = hits_3
        hits_max_matrix[0][2] = hits_5
        NDCG_max_matrix[0][0] = ndcg1
        NDCG_max_matrix[0][1] = ndcg3
        NDCG_max_matrix[0][2] = ndcg5
        MRR_max_matrix[0][0] = MRR
        epoch_max_matrix[0][0] = e
        patience_num_matrix[0][0] = 0
    else:
        patience_num_matrix[0][0] += 1
    return patience_num_matrix

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        # Perform linear transformation and split into heads
        query = self.q_linear(query).view(-1, self.n_heads, self.head_dim)
        key = self.k_linear(key).view(-1, self.n_heads, self.head_dim)
        value = self.v_linear(value).view(-1, self.n_heads, self.head_dim)

        # Transpose to get dimensions batch_size x n_heads x seq_len x head_dim
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Multiply attention scores with value
        out = torch.matmul(attention, value)

        # Transpose back and concatenate heads
        out = out.transpose(1, 2).contiguous().view(-1, self.d_model)

        # Linear layer for final output
        out = self.out_linear(out)

        return out


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm(x)
        # Multi-head attention
        attention_out = self.multihead_attention(x, x, x, mask)

        # Residual connection and normalization
        x = x + attention_out


        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Attention, self).__init__()
        self.layers = nn.ModuleList([AttentionLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

def construct_het_mat(rna_dis_mat, m_mat, d_mat):
    mat1 = np.hstack((d_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, m_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def compute_inf1(adj, L):
    c = 0.1
    W = torch.zeros(adj.shape).to(adj.device)
    D = (torch.diag(adj.sum(dim=1)) + 1e-15) ** (-1)
    for gamma in range(L):
        W += c * ((1 - c) ** gamma) * ((D @ adj) ** gamma)
    W = torch.mul(W, adj > 0)
    W = torch.softmax(W, dim=1)
    return W

def compute_inf(adj, L):
    c = 0.1
    W = torch.zeros_like(adj).to(adj.device)
    D = (torch.diag(adj.sum(dim=1)) + 1e-15) ** (-1)
    P = D @ adj
    for gamma in range(L):
        W += c * ((1 - c) ** gamma) * torch.matrix_power(P, gamma)
    W = W * (adj > 0)
    W = torch.softmax(W, dim=1)

    return W

def normalized_adj(adj):
    """
    输入：
        adj: NumPy 的稠密邻接矩阵（n x n），可以是 np.ndarray 或 torch.Tensor（转为 numpy）
    输出：
        norm_adj: PyTorch 的稠密 FloatTensor（n x n）
    功能：
        计算标准对称归一化邻接矩阵：
        norm_adj = D^{-1/2} * (A + I) * D^{-1/2}
    """

    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()


    adj = adj + np.eye(adj.shape[0])


    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0


    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return torch.from_numpy(norm_adj).float()
def getGosiR(asso_drug_cir):
    # calculate the r in GOsi Kerel
    nc = asso_drug_cir.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(asso_drug_cir[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def GIP_kernel(asso_drug_cir):
    # the number of row
    nc = asso_drug_cir.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(asso_drug_cir)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(asso_drug_cir[i, :] - asso_drug_cir[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def get_syn_sim(A, seq_sim, str_sim, mode):

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((A.shape[0], A.shape[0]))
    syn_d = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                # syn_c[i, j] = seq_sim[i, j]
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2

    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                # syn_d[i, j] = str_sim[i, j]
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_c, syn_d
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    # input_dim=128 name='gan_decoder' num_d=218
    def __init__(self, input_dim, name, num_d, dropout=0., act=torch.sigmoid,hid_out=256):
        super(InnerProductDecoder, self).__init__()
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_d = num_d
        self.w1 = nn.Linear(hid_out*2, 512)
        self.w2 = nn.Linear(hid_out*2, 512)
        # self.w1 = nn.Linear(input_dim, input_dim)
        self.att_drug = nn.Parameter(torch.rand(2), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(2), requires_grad=True)
        nn.init.xavier_uniform_(self.w1.weight)

    def forward(self, inputs, embd_cir, embd_drug,edge_label_index):
        # inputs = self.dropout(inputs)
        # embd_drug = self.dropout(embd_drug)
        # embd_cir = self.dropout(embd_cir)
        R = inputs[0:self.num_d, :]
        D = inputs[self.num_d:, :]
        R=self.w1(torch.cat((R,embd_drug),1))
        D=self.w2(torch.cat((D, embd_cir), 1))
        # D = D.T
        # x = R@D
        # x = torch.reshape(x, [-1])
        # outputs = self.act(x)
        # return outputs
        return (D[edge_label_index[0]] * R[edge_label_index[1]]).sum(dim=-1)

def get_metrics(real_score, predict_score):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
        recall_score, confusion_matrix
    true = real_score
    prob = predict_score
    # true = real_score
    pred = (np.array(prob) > 0.5).astype(int)

    auroc = roc_auc_score(true, prob)
    auprc = average_precision_score(true, prob)
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    recall = recall_score(true, pred)
    precision = precision_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp + 1e-6)
    return [auroc, auprc, f1, acc, recall, specificity, precision]

def calculate_loss(pred, pos_edge_idx, neg_edge_idx):
    pos_pred_socres = pred[pos_edge_idx[1], pos_edge_idx[0]]
    neg_pred_socres = pred[neg_edge_idx[1], neg_edge_idx[0]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0])))
    loss_fun=torch.nn.BCELoss(reduction='mean')
    pred_scores = pred_scores.cpu()
    # loss_fun=torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(pred_scores, true_labels)

def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges):
    pos_pred_socres = pred_mat[pos_edges[1], pos_edges[0]]
    neg_pred_socres = pred_mat[neg_edges[1], neg_edges[0]]
    predict_score = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0])))
    predict_score = predict_score.cpu()
    # np.save('SGT_pred(test).npy', pred_labels)
    # np.save('SGT_true(test).npy', true_labels)
    return true_labels, predict_score, get_metrics(true_labels, predict_score)

def sslLoss(proj_final_temporal, proj_global):

    proj_final_temporal = F.normalize(proj_final_temporal, dim=1)
    proj_global = F.normalize(proj_global, dim=1)


    posScore = torch.exp(torch.sum(proj_global * proj_final_temporal, dim=1) /1)


    sim_matrix = torch.matmul(proj_final_temporal, proj_global.t())
    negScore = torch.sum(torch.exp(sim_matrix / 1), dim=1)

    # ===== InfoNCE =====
    loss = -torch.log(posScore / (negScore + 1e-8) + 1e-8)

    return loss.mean()



def AMCLoss_v2(features,n_views,temp=0.1):
    device = features.device
    bsz = features.shape[0]

    # reshape
    features = F.normalize(features, dim=2)
    features = features.view(bsz * n_views, -1)

    # similarity
    sim = torch.matmul(features, features.t()) / temp

    # mask
    labels = torch.arange(bsz).repeat(n_views).to(device)
    # labels = torch.arange(bsz, device=device).repeat_interleave(n_views)
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # remove self
    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask

    # log_prob
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # mean over positives
    mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)

    loss = -mean_log_prob
    return loss.mean()

def sslLoss1(proj_final_temporal, proj_global):
    proj_final_temporal = F.normalize(proj_final_temporal, dim=1)
    proj_global = F.normalize(proj_global, dim=1)

    sim_matrix = torch.matmul(proj_final_temporal, proj_global.t()) / 0.1


    pos = torch.diag(sim_matrix)


    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

    loss = -pos + torch.logsumexp(sim_matrix, dim=1)

    return loss.sum()

def build_edge_index(adj_mat, threshold=0.5):
    edge_index = (adj_mat > threshold).nonzero(as_tuple=False).t()
    return edge_index

def build_edge_index_topk(sim_mat, k=20):
    N = sim_mat.shape[0]
    edge_list = []

    for i in range(N):
        idx = np.argsort(-sim_mat[i])[:k+1]
        for j in idx:
            if i != j:
                edge_list.append((i, j))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return edge_index

def build_edge_type(edge_index, num_drug):
    edge_type = []

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()

        if src < num_drug and dst < num_drug:
            edge_type.append(0)  # drug-drug
        elif src >= num_drug and dst >= num_drug:
            edge_type.append(1)  # microbe-microbe
        else:
            edge_type.append(2)  # drug-microbe

    return torch.tensor(edge_type, dtype=torch.long)


def parameters_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--seed', type=int, default=111, help='the random seeds')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.05, help="dropout rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--loss_gamma', type=float, default=0.7, help="balance coefficient")
    parser.add_argument('--hidden_size', type=int, default=256, help='model hidden dim')
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads of GAT')
    parser.add_argument('--patience', type=int, default=50, help='number of patience of early stopping mechanism')
    parser.add_argument('--hyperNum', default=512, type=int, help='number of hyper edges')
    parser.add_argument('--kfold', type=int, default=10, help='model hidden dim')
    parser.add_argument('--gnn_layer', default=5, type=int, help='number of gnn layers')
    parser.add_argument('--alpha', default=0.5, type=float, help='weight for final embedding')
    parser.add_argument('--ssl_weight', default=0.01, type=float, help='weight for ssl loss')
    parser.add_argument('--latdim', default=256, type=int, help='embedding size')
    parser.add_argument('--threshold', type=float, default=0.9)


    args = parser.parse_args()
    return args
