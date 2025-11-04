"""
PPI site prediction (sequence-based CNN model).

Exposes a single function:
    predict(id: str, data_dir: str = 'Original_Data', model_path: str = 'model/ppi_model10.h5',
            window_size: int = 19, threshold: float = 0.5 - 0.4) -> dict

Inputs
- id: Protein ID with chain (e.g., '1acbI'). Must match feature filenames in data_dir subfolders.
- data_dir: Base folder containing per-ID numpy features in subfolders 'pssm', 'hmm', 'dssp'.
- model_path: Path to the trained Keras model (.h5).
- window_size: Sliding window size used during training (default 19).
- threshold: Decision threshold to convert probabilities to 0/1 labels.

Outputs (dict)
- id: input id
- probs: list[float] per-residue binding probability
- labels: list[int] per-residue label (0/1) after thresholding

Notes
- This function assumes the feature files exist at
  {data_dir}/pssm/{id}.npy  (shape: L x 20)
  {data_dir}/hmm/{id}.npy   (shape: L x 20)
  {data_dir}/dssp/{id}.npy  (shape: L x 14)
- No training code is included.
"""
from __future__ import annotations
import os
from typing import Dict, List
import numpy as np

# Lazy import tensorflow to avoid overhead on module import
def _load_tf():
    import tensorflow as tf  # type: ignore
    return tf


def _windowing(features: np.ndarray, seq_len: int, w_size: int, start: int, stop: int) -> np.ndarray:
    """Apply centered sliding window over feature matrix.

    features: (L, F_total)
    Returns: (L, w_size, F_slice)
    """
    F_slice = stop - start
    out = np.zeros((seq_len, w_size, F_slice), dtype=np.float32)
    half = (w_size - 1) // 2
    for j in range(seq_len):
        for k in range(w_size):
            src = j + k - half
            if 0 <= src < seq_len:
                out[j, k, :] = features[src, start:stop]
    return out


def _load_features(id: str, data_dir: str) -> Dict[str, np.ndarray]:
    paths = {
        'pssm': os.path.join(data_dir, 'pssm', f'{id}.npy'),
        'hmm': os.path.join(data_dir, 'hmm', f'{id}.npy'),
        'dssp': os.path.join(data_dir, 'dssp', f'{id}.npy'),
    }
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing feature files for {id}: {', '.join(missing)} under {data_dir}")
    pssm = np.load(paths['pssm'])
    hmm = np.load(paths['hmm'])
    dssp = np.load(paths['dssp'])
    # Basic shape sanity
    L = pssm.shape[0]
    if hmm.shape[0] != L or dssp.shape[0] != L:
        raise ValueError(f'Feature length mismatch for {id}: L_pssm={L}, L_hmm={hmm.shape[0]}, L_dssp={dssp.shape[0]}')
    return {'pssm': pssm, 'hmm': hmm, 'dssp': dssp}


def predict(id: str,
            data_dir: str = 'Original_Data',
            model_path: str = 'model/ppi_model10.h5',
            window_size: int = 19,
            threshold: float = 0.5) -> Dict[str, List]:
    """Run PPI-site prediction for a single protein chain ID.

    Returns dict with probabilities and binary labels.
    """
    feats = _load_features(id, data_dir)
    L = feats['pssm'].shape[0]

    # The original model was trained with three inputs: PSSM (20), DSSP (14), HMM (20)
    f1 = _windowing(feats['pssm'], L, window_size, 0, feats['pssm'].shape[1])
    f2 = _windowing(feats['dssp'], L, window_size, 0, feats['dssp'].shape[1])
    f3 = _windowing(feats['hmm'],  L, window_size, 0, feats['hmm'].shape[1])

    tf = _load_tf()
    model = tf.keras.models.load_model(model_path)

    # Predict per-residue probabilities (flatten to L)
    probs = model.predict([f1, f2, f3], verbose=0)
    probs = np.squeeze(probs)
    probs = probs.astype(float).tolist()

    labels = [int(p >= threshold) for p in probs]
    return {'id': id, 'probs': probs, 'labels': labels}
