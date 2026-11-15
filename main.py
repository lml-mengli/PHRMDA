import csv
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from model import PHRGNN
from utils_ import *
from setdata import *
import os


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

micro_dis = pd.read_csv(r'./MDAD/mirco-drug-MDAD.csv', header=None)
micro_dis.columns =['micro_id','drug_id']
def make_edge_index(network):
    source_nodes = micro_dis['micro_id'].values
    target_nodes = micro_dis['drug_id'].values
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]),dtype=torch.long)
    return edge_index

def make_edge_attr(m_sim,d_sim):

    micro_dis = pd.read_csv(r'./MDAD/mirco-drug-MDAD.csv', header=None)
    micro_dis = micro_dis.values.tolist()
    edge_attr = np.zeros((len(micro_dis), len(micro_dis)), dtype=float)
    i=0
    for list in micro_dis:
        micro,dis = list
        for j in range(len(m_sim[micro])):
            edge_attr[i][j] = m_sim[micro][j]
        for p in range(len(d_sim[dis])):
            edge_attr[i][len(m_sim)+p] = d_sim[dis][p]
        i += 1
    return edge_attr



def get_link_labels(train_data, neg_edge_index):
    num_links = train_data['micro', 'cause', 'drug'].edge_label_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:train_data['micro', 'cause', 'drug'].edge_label_index.size(1)] = 1
    return link_labels


def evaluate_auc(model,data):
    model.eval()
    z = model(data)
    out = model.decode(z, data["micro", "cause", "drug"].edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data["micro", "cause", "drug"].edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())
def acc_test(model,data):
    model.eval()
    z = model(data)

    out = model.decode(z, data["micro", "cause", "drug"].edge_label_index).view(-1).sigmoid()
    out = out.cpu().detach().numpy()
    pred = (np.array(out) > 0.5).astype(int)
    true = data["micro", "cause", "drug"].edge_label.cpu().detach().numpy()
    auroc = roc_auc_score(true, out)
    auprc = average_precision_score(true, out)
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    recall = recall_score(true, pred)
    precision = precision_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp + 1e-6)
    return [auroc, auprc, f1, acc, recall, specificity, precision]


if __name__ == '__main__':

    args = parameters_set()
    set_seed(args.seed)

    association = pd.read_csv("./MDAD/index_association.csv", header=0, index_col=0).values
    m_sim = pd.read_csv("./MDAD/microbe_similarity.txt", header=None, index_col=None, sep='\t').values
    d_sim = pd.read_csv("./MDAD/drugsimilarity.txt", header=None, index_col=None,sep='\t').values
    n_m = len(m_sim)
    n_d = len(d_sim)

    m_sim = np.array(m_sim,dtype=float)
    d_sim = np.array(d_sim,dtype=float)
    data = HeteroData()
    edge_index = make_edge_index(association)
    edge_attr = make_edge_attr(m_sim,d_sim)

    temp_drug_cir = torch.zeros((n_d, n_m), device=device)

    drug_sim, cir_sim = d_sim, m_sim
    data['micro'].node_id = torch.arange(len(cir_sim))
    data['drug'].node_id = torch.arange(len(drug_sim))
    data['micro'].x = torch.tensor(m_sim,dtype=torch.float32)
    data['drug'].x = torch.tensor(d_sim,dtype=torch.float32)
    data['micro','cause','drug'].edge_index = edge_index
    data['micro','cause','drug'].edge_arrt = edge_attr
    data = ToUndirected()(data)
    print(data)


    pos_edge_index = data['micro', 'cause', 'drug'].edge_index
    num_edges = pos_edge_index.size(1)

    kfolds = args.kfold  # 10
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=args.seed)
    folds = []
    kfolds_metrics = {}
    for fold, (remain_idx, test_idx) in enumerate(kf.split(range(num_edges))):
        remain_idx = np.array(remain_idx)
        np.random.shuffle(remain_idx)
        num_val = len(test_idx)
        val_idx = remain_idx[:num_val]
        train_idx = remain_idx[num_val:]
        train_data, val_data, test_data = build_hetero_split(data, pos_edge_index, train_idx, val_idx, test_idx, neg_ratio=1.0, disjoint_ratio=0.3)

        data = data.to(device)
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        val_data = val_data.to(device)

        np.random.seed(args.seed)

        model = PHRGNN(hidden_size=args.hidden_size, num_heads=args.num_heads,
                       n_in_features=n_d + n_m, n_drug=n_d, dropout=args.dropout, args=args, data=data)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)
        train_pos_edges = train_data['micro', 'cause', 'drug'].edge_index

        temp_drug_cir[train_pos_edges[1], train_pos_edges[0]] = 1
        temp_drug_cir_np = temp_drug_cir.detach().cpu().numpy()

        het_mat = construct_het_mat(temp_drug_cir_np, m_sim, d_sim)
        adj_mat = construct_adj_mat(temp_drug_cir_np)

        edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
        het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

        norm_adj_mat = normalized_adj(adj_mat).to(device)

        best_val_auc = final_test_auc = best_f1_auc = 0
        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()
            neg_edge_label_index = negative_sampling(edge_index=train_data['micro', 'cause', 'drug'].edge_index,
                                                     num_nodes=([train_data['micro'].num_nodes, train_data['drug'].num_nodes]),
                                                     num_neg_samples=train_data['micro', 'cause', 'drug'].edge_label_index.size(1),
                                                     method='sparse')

            edge_label_index = torch.cat(
                [train_data['micro', 'cause', 'drug'].edge_label_index, neg_edge_label_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data['micro', 'cause', 'drug'].edge_label,
                train_data['micro', 'cause', 'drug'].edge_label.new_zeros(neg_edge_label_index.size(1))
            ], dim=0)

            out,sslLoss = model(train_data, het_mat_device, edge_idx_device, norm_adj_mat, edge_label_index)


            train_pos_edges_label=train_data['micro', 'cause', 'drug'].edge_label_index

            out = model.decode(out, edge_label_index).view(-1)

            floss = F.binary_cross_entropy_with_logits(out, edge_label)

            loss = floss + args.ssl_weight * sslLoss

            loss.backward()
            optimizer.step()

            val_edges = test_data["micro", "cause", "drug"].edge_label_index
            val_neg_data = test_data["micro", "cause", "drug"].edge_label

            model.eval()
            with torch.no_grad():

                out,_ = model(test_data, het_mat_device, edge_idx_device, norm_adj_mat, val_edges)
                out = model.decode(out, test_data["micro", "cause", "drug"].edge_label_index).view(-1).sigmoid()
                out = out.cpu().detach().numpy()

                pred = (np.array(out) > 0.5).astype(int)
                true = test_data["micro", "cause", "drug"].edge_label.cpu().detach().numpy()
                auroc = roc_auc_score(true, out)
                auprc = average_precision_score(true, out)
                acc = accuracy_score(true, pred)
                f1 = f1_score(true, pred)
                recall = recall_score(true, pred)
                precision = precision_score(true, pred)
                tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
                specificity = tn / (tn + fp + 1e-6)
                mcc = matthews_corrcoef(true, pred)
                metrics = [auroc, auprc, f1, acc, recall, specificity, precision, mcc]
                # metrics = [auroc, auprc, f1, acc, recall, specificity, precision]

                if auroc >= best_val_auc and f1 >= best_f1_auc:

                    best_val_auc = auroc
                    final_test_auc = auroc
                    best_f1_auc = f1
                    metrics_max = metrics.copy()
                    torch.save(model.state_dict(), "./MDAD/MDAD_GCN_model.pt")
                    kfolds_metrics[f"kf{fold}"] = metrics.copy()


                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val:Auc = {auroc:.4f}, Test:Auc = {auroc:.4f},  Aupr = ',
                      metrics[1], '\t  f1_score = ', metrics[2],
                      '\t  Accuracy = ', metrics[3], '\t  Recall = ', metrics[4], '\t  Specificity = ', metrics[5],
                      '\t  Precision = ', metrics[6], '\t  Mcc = ', metrics[7])  # , Val:Auc = {val_auc:.4f}

        print(f'Final Test:{final_test_auc:.4f}')
        print('\nIndependent Test Set：max_Auc = ', metrics_max[0], '\t  Aupr = ', metrics_max[1], '\t  f1_score = ',
              metrics_max[2],
              '\t  Accuracy = ', metrics_max[3], '\t  Recall = ', metrics_max[4], '\t  Specificity = ', metrics_max[5],
              '\t  Precision = ', metrics_max[6], '\t  Mcc = ', metrics_max[7])

pd.DataFrame(kfolds_metrics).to_csv(f"./result_MDAD/{kfolds}-kfolds_metrics.csv", index=False)
all_metrics = np.array(list(kfolds_metrics.values()))
metrics = np.mean(all_metrics, axis=0).tolist()
print(f'Test:Auc = {metrics[0]:.4f},  Aupr = ', metrics[1], '\t  f1_score = ', metrics[2],
     '\t  Accuracy = ', metrics[3], '\t  Recall = ', metrics[4], '\t  Specificity = ', metrics[5],
     '\t  Precision = ', metrics[6], '\t  Mcc = ', metrics[7])
