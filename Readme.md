Enhanced PPI-site Prediction

This repository now provides inference-only entry points for two models:

- Sequence CNN model: `ppi_predict.py`
- Graph GNN model (GraphPPIS): `graph_predict.py`

Most of the training code is included in the experimental code folder. Each module above exposes a single `predict(...)` function for inference.

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

## Feature generation toolchain

To compute the residue-level features required by the predictors you need three external tools and their databases. If you already have the `.npy` files under `Original_Data/` you can skip this section.

Required software:

1. DSSP – assigns secondary structure and solvent accessibility.
2. BLAST+ (PSI-BLAST) – builds PSSM profiles using iterative sequence searches.
3. HH-suite (hhblits) – generates HMM/HHM profiles capturing remote homology.

Required databases:

* UniRef90 (for PSI-BLAST) – make a BLAST database with `makeblastdb`.
* Uniclust30 (for hhblits) – prebuilt HMM cluster database.

### Building databases & paths

1. Build UniRef90 BLAST database:
	 ```bash
	 makeblastdb -in uniref90.fasta -dbtype prot -parse_seqids -out uniref90
	 ```
2. Obtain / build Uniclust30 following the HH-suite instructions.
3. Install DSSP (e.g. via package manager or source) and ensure the executable `mkdssp` is on disk.
4. Set the paths in `Utilis.py` (or adapt in your own script):
	 - `UR90` → path to UniRef90 FASTA (base name used by PSI-BLAST)
	 - `HHDB` → path prefix to Uniclust30 HMM database
	 - `PSIBLAST` → full path to `psiblast` executable
	 - `HHBLITS` → full path to `hhblits` executable
	 - `dssp` → full path to `mkdssp` executable

### Output file expectations

After running your feature extraction pipeline each protein chain ID (e.g. `1acbI`) should have:

```
Original_Data/
	pssm/1acbI.npy    (L x 20)  # normalized PSSM
	hmm/1acbI.npy     (L x 20)  # HHM/HHblits profile
	dssp/1acbI.npy    (L x 14)  # DSSP attributes
	dismap/1acbI.npy   (L x L)   # distance map (for graph model only)
```

If any file is missing, the corresponding predictor will raise an exception telling you which one to generate.

### Benchmarking feature generation

You can measure how long PSI-BLAST and HH-suite take on your machine by prepending `time` to each command. Replace the paths below with your own.

**PSI-BLAST (PSSM generation):**

```bash
time psiblast -db /path/to/uniref90 -num_iterations 3 -num_alignments 1 \
  -num_threads 2 -query train/1acbI.fa -out train/1acbI.bla \
  -out_ascii_pssm train/pssm/1acbI.pssm
```

**HH-suite / hhblits (HHM generation):**

```bash
time hhblits -i train/1acbI.fa -ohhm train/hhm/1acbI.hhm \
  -oa3m train/1acbI.a3m -d /path/to/uniclust30 \
  -v 0 -maxres 40000 -cpu 6 -Z 0 -o train/1acbI.hhr
```

To see how parallelism affects runtime, adjust `-num_threads` (PSI-BLAST) or `-cpu` (hhblits).

### Fast vs full graph features

The original GraphPPIS framework supports a fast mode (DSSP + BLOSUM) and a full mode (PSSM + HMM + DSSP). This repository’s `graph_predict.py` implements the full feature mode (54-dim vector per residue). To create a fast-mode variant you would concatenate BLOSUM62 (20) + DSSP (14) = 34 features and load the appropriate weights.

