import os
import torch 
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader
from Grapghnn import *  # Assuming you have a module named GraphPPIS
from Bio import PDB
import gzip

app = Flask(__name__)

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
model_path = "./Model/"

# GraphPPIS parameters
MAP_CUTOFF = 14
HIDDEN_DIM = 256
LAYER = 8
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True  # From GCNII

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_chain_atoms(pdb_file, chain_id):
    # Parse the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chain_atoms = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    for atom in residue:
                        chain_atoms.append({
                            'atom_name': atom.get_name(),
                            'residue_name': residue.get_resname(),
                            'residue_id': '{}'.format(residue.get_id()[1]),
                            'x': str(round(atom.coord[0],3)),
                            'y': str(round(atom.coord[1], 3)),
                            'z': str(round(atom.coord[2], 3))
                        })
    return chain_atoms

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    protein_id = data.get('protein_id')
    chain_identifier = data.get('chain_identifier')
    data_path = './Data/'  # Specify your data path here
    pdbs_path = './PDBs/'

    mapping_dict= {'1':'A','2':'B','3':'C','4':'D','5':'E'}

    if chain_identifier in mapping_dict:
        chain_identifier = mapping_dict[chain_identifier]

    id = protein_id + chain_identifier

    ProtName_train, seqs_train, y_train = parse_fasta('./Data/Train_335.fa')
    ProtName_test, seqs_test, y_test = parse_fasta('./test/Test_60.fa')
    ProtName_test315, seqs_test315, y_test315 = parse_fasta('./test/Test_315.fa')

    # concat train and test data
    ProtName = ProtName_train + ProtName_test + ProtName_test315
    seqs = seqs_train + seqs_test + seqs_test315
    y = y_train + y_test + y_test315

    # Search for the protein in the dataset
    if id not in ProtName:
        return jsonify({'error': 'Not found'}), 404

    # load the requested protein
    idx = ProtName.index(id)
    protein = ProtName[idx]
    amino_acids = seqs[idx]
    y_id = y[idx]

    # Load the test data
    test_dataframe = pd.DataFrame(list(zip([protein], [y_id])), columns=['ID', 'label'])

    # Run the model
    test_pred = test(test_dataframe, data_path, 'normal')
    test_pred = np.array(test_pred)
    y_pred = np.where(test_pred >= 0.273, 1, 0)

    # set precision for y_preds to the second number



    # Prepare the response
    response = {
        'protein_id': protein_id,
        'chain_identifier': chain_identifier,
        'amino_acids': []
    }

    for idx, amino_acid in enumerate(amino_acids):
        response['amino_acids'].append({
            'amino_acid': amino_acid,
            'position': '{}'.format(idx + 1),
            'ratio': str((round(test_pred[idx],3))),
            'IsBindingSite': '{}'.format(y_pred[idx])
        })

    # Extract chain atoms from pdb.gz
    pdb_file_path = os.path.join(pdbs_path, f"{protein_id}.pdb.gz")
    if os.path.isfile(pdb_file_path):
        with gzip.open(pdb_file_path, 'rt') as pdb_file:
            chain_atoms = extract_chain_atoms(pdb_file, chain_identifier)
            response['chain_atoms'] = chain_atoms
    else:
        response['chain_atoms'] = 'PDB file not found'

    return jsonify(response)

def test(test_dataframe, data_path, mode):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe, data_path, mode), batch_size=BATCH_SIZE, shuffle=False)

    INPUT_DIM = 54
    GraphPPIS_model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    GraphPPIS_model.load_state_dict(torch.load("./API/GraphPPIS_normal.pkl", map_location=device))

    test_pred = evaluate(GraphPPIS_model, test_loader)

    return test_pred

if __name__ == '__main__':
    app.run(debug=True)
