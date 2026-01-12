import numpy as np
import tensorflow as tf
import torch
from typing import List, Tuple, Optional, Literal, Union

Mode = Literal["over", "under", "both"]

class BalancedBatchGenerator(tf.keras.utils.Sequence):
    """
    Balanced mini-batches for binary classification with multi-input X (list of arrays).
    - Fixed batch size
    - Enforces target_pos_fraction (default 0.5)
    - Shuffles safely using shared indices
    - batched splits is done per training, validation data separately (no data leakage)
    """
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=0, target_pos_fraction=0.5):
        assert isinstance(X, (list, tuple)) and len(X) > 0, "X must be a list/tuple of input arrays"
        self.X = X
        self.y = np.asarray(y)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.target_pos = int(round(self.batch_size * float(target_pos_fraction)))
        self.target_neg = self.batch_size - self.target_pos

        self.pos_pool = np.where(self.y == 1)[0]
        self.neg_pool = np.where(self.y == 0)[0]
        if len(self.pos_pool) == 0 or len(self.neg_pool) == 0:
            raise ValueError("Need at least one sample from each class for balanced batching.")

        self.steps = int(np.ceil(len(self.y) / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        pos_idx = self.rng.choice(self.pos_pool, size=self.target_pos, replace=(len(self.pos_pool) < self.target_pos))
        neg_idx = self.rng.choice(self.neg_pool, size=self.target_neg, replace=(len(self.neg_pool) < self.target_neg))

        batch_idx = np.concatenate([pos_idx, neg_idx])

        if self.shuffle:
            p = self.rng.permutation(len(batch_idx))
            batch_idx = batch_idx[p]

        X_batch = [x[batch_idx] for x in self.X]
        y_batch = self.y[batch_idx]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)


from tensorflow.keras.callbacks import Callback

class ReduceLROnValLoss(Callback):
    def __init__(self, monitor='val_loss', threshold=0.39, factor=0.1, min_lr=0.00008, verbose=1):
        super(ReduceLROnValLoss, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_lr = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get(self.monitor)

        if current_val_loss is not None and current_val_loss < self.threshold:
            if self.best_lr is None:
                self.best_lr = self.model.optimizer.lr.numpy()
            new_lr = max(self.min_lr, self.best_lr * self.factor)
            self.model.optimizer.lr.assign(new_lr)
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: Reduced learning rate to {new_lr:.6f} based on validation loss.')
            self.best_lr = new_lr


def balance_data(
    X: List[np.ndarray],
    y: np.ndarray,
    *,
    mode: Mode = "both",
    pos_fraction: Optional[float] = None,
    pos_to_neg_ratio: Optional[float] = None,
    n_samples: Optional[int] = None,
    replace_if_needed: bool = True,
    seed: Optional[int] = 42,
    return_indices: bool = False,
) -> Union[Tuple[List[np.ndarray], np.ndarray], Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:
    """
    Balance a binary dataset by (over)sampling positives and/or (under)sampling negatives.

    Parameters
    ----------
    X : list of np.ndarray
        Multi-input features. Each X[k] must have shape (N, ...).
    y : np.ndarray
        Binary labels of shape (N,), values in {0,1}.
    mode : {"over", "under", "both"}
        - "over": increase minority class by sampling WITH replacement from it
        - "under": reduce majority class by sampling WITHOUT replacement from it
        - "both": do both to reach the requested target composition
    pos_fraction : float, optional
        Desired fraction of positives in the output, in (0,1). Example: 0.5 for balanced.
        Mutually exclusive with pos_to_neg_ratio.
    pos_to_neg_ratio : float, optional
        Desired ratio pos:neg in the output. Example: 1.0 means equal counts.
        Mutually exclusive with pos_fraction.
    n_samples : int, optional
        Total number of samples in the balanced output. If None, uses the current N.
        For mode="over", output may exceed N if needed (unless you set n_samples).
    replace_if_needed : bool
        If True, allows replacement when a class doesn't have enough samples.
    seed : int or None
        RNG seed for reproducibility. None => non-deterministic.
    return_indices : bool
        If True, also return the selected indices (useful for debugging/repro notes).

    Returns
    -------
    X_bal : list of np.ndarray
    y_bal : np.ndarray
    idx_bal : np.ndarray (optional)
        Indices used to construct the balanced dataset.
    """
    if (pos_fraction is None) == (pos_to_neg_ratio is None):
        raise ValueError("Provide exactly one of pos_fraction or pos_to_neg_ratio.")

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D of shape (N,)")

    N = len(y)
    if not all(len(x) == N for x in X):
        raise ValueError("All X inputs must have the same first dimension as y.")

    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must be present to balance data.")

    if pos_fraction is not None:
        if not (0.0 < pos_fraction < 1.0):
            raise ValueError("pos_fraction must be in (0,1).")
        if n_samples is None:
            n_samples = N
        target_pos = int(round(n_samples * pos_fraction))
        target_neg = n_samples - target_pos
    else:
        if pos_to_neg_ratio <= 0:
            raise ValueError("pos_to_neg_ratio must be > 0.")
        if n_samples is None:
            target_neg = n_neg
            target_pos = int(round(target_neg * pos_to_neg_ratio))
        else:
            r = float(pos_to_neg_ratio)
            target_neg = int(round(n_samples / (1.0 + r)))
            target_pos = n_samples - target_neg

    def sample(pool: np.ndarray, k: int, replace: bool) -> np.ndarray:
        if k <= 0:
            return np.empty((0,), dtype=int)
        if (not replace) and (k > len(pool)):
            if not replace_if_needed:
                raise ValueError(f"Not enough samples in pool ({len(pool)}) to sample {k} without replacement.")
            replace = True
        return rng.choice(pool, size=k, replace=replace)

    if mode == "over":
        chosen_neg = sample(neg_idx, min(target_neg, n_neg), replace=False)
        chosen_pos = sample(pos_idx, target_pos, replace=(target_pos > n_pos))
    elif mode == "under":
        chosen_pos = sample(pos_idx, min(target_pos, n_pos), replace=False)
        chosen_neg = sample(neg_idx, target_neg, replace=False)
    elif mode == "both":
        chosen_pos = sample(pos_idx, target_pos, replace=(target_pos > n_pos))
        chosen_neg = sample(neg_idx, target_neg, replace=False)
        if target_neg > n_neg:
            chosen_neg = sample(neg_idx, target_neg, replace=True)
    else:
        raise ValueError("mode must be one of: 'over', 'under', 'both'")

    idx_bal = np.concatenate([chosen_pos, chosen_neg])
    idx_bal = rng.permutation(idx_bal)

    X_bal = [x[idx_bal] for x in X]
    y_bal = y[idx_bal]

    if return_indices:
        return X_bal, y_bal, idx_bal
    return X_bal, y_bal




def train(model, device, train_loader, optimizer, criterion, epoch, max_grad_norm=5.0):
    """
    Training loop for graph-based model.
    
    Note: Optional class weights can be passed to criterion if needed for class imbalance.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch_idx, (node_features, adjacency_matrix, labels) in enumerate(train_loader):

        node_features = node_features.to(device)
        adjacency_matrix = adjacency_matrix.to(device)
        labels = labels.to(device).reshape(-1).long()

        if node_features.dim() == 3 and node_features.size(0) == 1:
            node_features = node_features[0]
        if adjacency_matrix.dim() == 3 and adjacency_matrix.size(0) == 1:
            adjacency_matrix = adjacency_matrix[0]

        if node_features.dim() != 2 or adjacency_matrix.dim() != 2:
            print(f"Invalid shapes batch {batch_idx}: X={tuple(node_features.shape)}, A={tuple(adjacency_matrix.shape)}")
            continue

        optimizer.zero_grad(set_to_none=True)

        output = model(node_features, adjacency_matrix)

        if not torch.isfinite(output).all():
            print(f"Non-finite output at batch {batch_idx}")
            continue

        loss = criterion(output, labels)

        if not torch.isfinite(loss):
            print(f"Non-finite loss at batch {batch_idx} epoch {epoch+1}")
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if not torch.isfinite(grad_norm):
            print(f"Non-finite grad norm at batch {batch_idx}")
            continue

        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg = total_loss / count if count else float("inf")
    print(f"Epoch {epoch+1}, Avg Loss: {avg:.4f}, Batches used: {count}")




