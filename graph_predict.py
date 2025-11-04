"""
Graph-based PPI site prediction (GraphPPIS model).

Exposes a single function:
    predict(id: str, data_dir: str = 'Original_Data', weights_path: str = 'model/GraphPPIS_normal.pkl',
            threshold: float = 0.273) -> dict

Inputs
- id: Protein ID with chain (e.g., '1acbI'). Must match feature/graph filenames in data_dir.
- data_dir: Base folder containing numpy arrays in subfolders:
    pssm/{id}.npy (L x 20), hmm/{id}.npy (L x 20), dssp/{id}.npy (L x 14), dismap/{id}.npy (L x L)
- weights_path: Path to the trained PyTorch state_dict for GraphPPIS.
- threshold: Decision threshold to convert probabilities to 0/1 labels.

Outputs (dict)
- id: input id
- probs: list[float] per-residue binding probability
- labels: list[int] per-residue label (0/1) after thresholding

No training code is included.
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

SEED = 2020
np.random.seed(SEED)
_ = torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (must match training)
INPUT_DIM = 54  # 20+20+14
HIDDEN_DIM = 256
LAYER = 8
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True
NUM_CLASSES = 2


def _normalize(mx: np.ndarray) -> np.ndarray:
    rowsum = np.array(mx.sum(1))
    r_inv = (np.abs(rowsum) ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv @ mx @ r_mat_inv


def _load_features_and_graph(id: str, data_dir: str) -> Dict[str, np.ndarray]:
    paths = {
        'pssm': os.path.join(data_dir, 'pssm', f'{id}.npy'),
        'hmm': os.path.join(data_dir, 'hmm', f'{id}.npy'),
        'dssp': os.path.join(data_dir, 'dssp', f'{id}.npy'),
        'dismap': os.path.join(data_dir, 'dismap', f'{id}.npy'),
    }
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing files for {id}: {', '.join(missing)} under {data_dir}")
    pssm = np.load(paths['pssm']).astype(np.float32)
    hmm = np.load(paths['hmm']).astype(np.float32)
    dssp = np.load(paths['dssp']).astype(np.float32)
    dis = np.load(paths['dismap']).astype(np.float32)

    L = pssm.shape[0]
    if hmm.shape[0] != L or dssp.shape[0] != L or dis.shape[0] != L or dis.shape[1] != L:
        raise ValueError('Feature/graph dimension mismatch for {id}')

    x = np.concatenate([pssm, hmm, dssp], axis=1)  # (L, 54)
    adj = _normalize(dis)
    return {'x': x, 'adj': adj}


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super().__init__()
        self.variant = variant
        self.in_features = 2 * in_features if self.variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = min(1, np.log(lamda / l + 1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class DeepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super().__init__()
        self.convs = nn.ModuleList([GraphConvolution(nhidden, nhidden, residual=True, variant=variant) for _ in range(nlayers)])
        self.fcs = nn.ModuleList([nn.Linear(nfeat, nhidden), nn.Linear(nhidden, nclass)])
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        layers0 = []
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        layers0.append(h)
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h, adj, layers0[0], self.lamda, self.alpha, i + 1))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fcs[-1](h)
        return h


class GraphPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super().__init__()
        self.gcn = DeepGCN(nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant)

    def forward(self, x, adj):
        x = x.float()
        return self.gcn(x, adj)


@torch.no_grad()
def _inference(model: nn.Module, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    model.eval()
    x_t = torch.from_numpy(x)
    adj_t = torch.from_numpy(adj)
    logits = model(x_t, adj_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()  # (L, 2)
    return probs[:, 1]  # binding probability


def predict(id: str,
            data_dir: str = 'Original_Data',
            weights_path: str = 'model/GraphPPIS_normal.pkl',
            threshold: float = 0.273) -> Dict[str, List]:
    """Run GraphPPIS prediction for a single protein chain ID.

    Returns dict with probabilities and binary labels.
    """
    data = _load_features_and_graph(id, data_dir)
    model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)

    probs = _inference(model, data['x'], data['adj'])
    probs_list = probs.astype(float).tolist()
    labels = [int(p >= threshold) for p in probs_list]
    return {'id': id, 'probs': probs_list, 'labels': labels}
