Enhanced PPI-site Prediction

This repository now provides inference-only entry points for two models:

- Sequence CNN model: `ppi_predict.py`
- Graph GNN model (GraphPPIS): `graph_predict.py`

No training code is included in these modules; each exposes a single `predict(...)` function.

Usage

Python 3.8+ with packages numpy, tensorflow (for PPI CNN) and torch (for GraphPPIS) is required.

1) PPI (sequence) model

The PPI model expects precomputed per-residue features in `Original_Data/`:

- `pssm/{ID}.npy`  (L x 20)
- `hmm/{ID}.npy`   (L x 20)
- `dssp/{ID}.npy`  (L x 14)

Weights: `model/ppi_model10.h5`

Example

```python
from ppi_predict import predict
res = predict('1acbI', data_dir='Original_Data', model_path='model/ppi_model10.h5', threshold=0.5)
print(res['labels'][:10])  # first 10 residues
```

2) Graph (GraphPPIS) model

The GraphPPIS model expects the same residue features plus a distance map:

- `pssm/{ID}.npy`  (L x 20)
- `hmm/{ID}.npy`   (L x 20)
- `dssp/{ID}.npy`  (L x 14)
- `dismap/{ID}.npy` (L x L)

Weights: `model/GraphPPIS_normal.pkl`

Example

```python
from graph_predict import predict
res = predict('1acbI', data_dir='Original_Data', weights_path='model/GraphPPIS_normal.pkl', threshold=0.273)
print(res['labels'][:10])
```

Chain extraction helper

For preparing inputs from PDB files, use `get_chain2.py` (recommended). It can extract a specific chain and now writes both a filtered PDB and a FASTA file for that chain.

Notes

- The legacy `API/` Flask code is deprecated in favor of the standalone predict functions above.
- Feature computation utilities remain in `Utilis.py` for reference, but are not invoked by the new predictors.
- Ensure your IDs (e.g., `1acbI`) match the filenames in `Original_Data`.
