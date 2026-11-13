from sklearn.model_selection import KFold
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import negative_sampling


def build_hetero_split(
    data,
    pos_edge_index,
    train_pos_idx,
    val_idx,
    test_pos_idx,
    neg_ratio=1.0,
    disjoint_ratio=0.3
):
    """
    构造一折 train_data / test_data
    """

    num_micro = data['micro'].num_nodes
    num_drug = data['drug'].num_nodes

    # ========= 1. 取正样本 =========
    train_pos_edge = pos_edge_index[:, train_pos_idx]
    val_pos_edge = pos_edge_index[:, val_idx]
    test_pos_edge = pos_edge_index[:, test_pos_idx]

    shared_mpv_edge = torch.cat([train_pos_edge, val_pos_edge], dim=1)
    shared_mpt_edge = train_pos_edge

    # ========= 2. 在 train 内部做 7:3 =========
    num_train_edges = train_pos_edge.size(1)
    perm = torch.randperm(num_train_edges)

    sup_size = int(num_train_edges * disjoint_ratio)
    sup_idx = perm[:sup_size]  # 30% 监督
    mp_idx = perm[sup_size:]  # 70% 传播

    train_mp_edge = train_pos_edge[:, mp_idx]  # edge_index
    train_sup_pos = train_pos_edge[:, sup_idx]  # 正监督边

    # ========= 2. 负采样 =========
    num_train_neg = int(train_pos_edge.size(1) * neg_ratio)
    num_val_neg = int(val_pos_edge.size(1) * neg_ratio)
    num_test_neg  = int(test_pos_edge.size(1) * neg_ratio)

    train_neg_edge = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(num_micro, num_drug),
        num_neg_samples=num_train_neg,
        method='sparse'
    )

    val_neg_edge = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(num_micro, num_drug),
        num_neg_samples=num_val_neg,
        method='sparse'
    )

    test_neg_edge = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(num_micro, num_drug),
        num_neg_samples=num_test_neg,
        method='sparse'
    )

    # ========= 3. edge_label_index / label =========
    train_edge_label_index = train_sup_pos
    train_edge_label =  torch.ones(train_sup_pos.size(1))

    test_edge_label_index = torch.cat([test_pos_edge, test_neg_edge], dim=1)
    test_edge_label = torch.cat([
        torch.ones(test_pos_edge.size(1)),
        torch.zeros(test_neg_edge.size(1))
    ])

    val_edge_label_index= torch.cat([val_pos_edge, val_neg_edge], dim=1)
    val_edge_label = torch.cat([
        torch.ones(val_pos_edge.size(1)),
        torch.zeros(val_neg_edge.size(1))
    ])

    # ========= 4. 构造 train_data =========
    train_data = HeteroData()
    train_data['micro'].x = data['micro'].x
    train_data['micro'].node_id = data['micro'].node_id
    train_data['drug'].x = data['drug'].x
    train_data['drug'].node_id = data['drug'].node_id

    train_data['micro', 'cause', 'drug'].edge_index = train_mp_edge
    train_data['micro', 'cause', 'drug'].edge_label_index = train_edge_label_index
    train_data['micro', 'cause', 'drug'].edge_label = train_edge_label

    train_data = ToUndirected()(train_data)

    # ========= 5. 构造 val_data =========
    val_data = HeteroData()
    val_data['micro'].x = data['micro'].x
    val_data['micro'].node_id = data['micro'].node_id
    val_data['drug'].x = data['drug'].x
    val_data['drug'].node_id = data['drug'].node_id

    val_data['micro', 'cause', 'drug'].edge_index = shared_mpt_edge
    val_data['micro', 'cause', 'drug'].edge_label_index = val_edge_label_index
    val_data['micro', 'cause', 'drug'].edge_label = val_edge_label
    val_data = ToUndirected()(val_data)
    # ========= 6. 构造 test_data =========
    test_data = HeteroData()
    test_data['micro'].x = data['micro'].x
    test_data['micro'].node_id = data['micro'].node_id
    test_data['drug'].x = data['drug'].x
    test_data['drug'].node_id = data['drug'].node_id

    test_data['micro', 'cause', 'drug'].edge_index = shared_mpv_edge #!!!!!!!!train_pos_edge,test_pos_edge
    test_data['micro', 'cause', 'drug'].edge_label_index = test_edge_label_index
    test_data['micro', 'cause', 'drug'].edge_label = test_edge_label

    test_data = ToUndirected()(test_data)

    return train_data,val_data, test_data

def ten_fold_cv(data, n_splits=10, seed=42):

    pos_edge_index = data['micro', 'cause', 'drug'].edge_index
    num_edges = pos_edge_index.size(1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(num_edges))):
        train_data, test_data = build_hetero_split(
            data,
            pos_edge_index,
            train_idx,
            test_idx,
            neg_ratio=1.0
        )
        folds.append((train_data, test_data))

    return train_data, test_data
