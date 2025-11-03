from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import gzip
import os

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "ASX": "B", "GLX": "Z", "XAA": "X",
    "NME": "M", "ACE": "A", "UNK": "X"
}

def extract_chain(pdb_file, chain_id, output_pdb):
    parser = PDB.PDBParser(QUIET=True)

    # Open the PDB file (handle gzip if necessary)
    if pdb_file.endswith(".gz"):
        with gzip.open(pdb_file, 'rt') as f:
            structure = parser.get_structure("structure", f)
    else:
        structure = parser.get_structure("structure", pdb_file)

    model = structure[0]  # Get the first model

    # convert chain IDs to alphabetical from 1,2,3 to A,B,C (if existed)
    # for chain in model:
    #     id = chain.get_id()
    #     if id.isdigit():
    #         id = chr(ord('A') + int(id) - 1)
    #         chain.id = id
    # convert chain model from A to 0 , B to 1 , etc
    
    id = chain_id
    if id.isalpha():
        id = ord(id.upper()) - ord('A')
    chain_id = id 
    # Ensure the chain exists
    # if chain_id not in model:
    #     print(f"Chain {chain_id} not found in {pdb_file}")
    #     return

    chains = [chain for chain in model]
    if chain_id >= len(chains):
        print(f"Chain {chain_id} not found in {pdb_file}")
        return
    chain = chains[chain_id]

    # Collect the sequence in a list (to write to FASTA later)
    seq_list = []

    # Remove unwanted atoms (Hydrogens) and replace certain residues
    io = PDB.PDBIO()
    io.set_structure(structure)

    class SelectChainAndAtoms(PDB.Select):
        def __init__(self, selected_chain):
            super().__init__()
            self.selected_chain = selected_chain

        def accept_chain(self, chain):
            # Accept the chain if it matches the selected chain from the chains list
            return chain == self.selected_chain

        def accept_atom(self, atom):
            # Exclude hydrogen atoms (names start with 'H')
            return not atom.get_id().startswith('H')

        def accept_residue(self, residue):
            # Replace MSE with MET
            if residue.get_resname() == "MSE":
                residue.resname = "MET"
            
            # Collect the one-letter code for residues (skip HETATMs)
            if residue.id[0] == ' ':  # Only standard residues
                resname = residue.get_resname()
                if resname in THREE_TO_ONE:
                    aa = THREE_TO_ONE[resname]
                    seq_list.append(aa)
                else:
                    # If the residue is not in the dictionary, append 'X'
                    seq_list.append('X')
            return True

    # Save the new PDB file with the selected chain and modified atoms
    io.save(output_pdb, select=SelectChainAndAtoms(chain))
    print(f"Saved chain {chain_id} to {output_pdb}")


def main():
    pdb_dir = "./data_ext/"  # Modify this to your PDB directory
    pdb_id_chain = "7NEGB"  # Example PDB ID and chain (e.g., 3mcbA)

    pdb_id = pdb_id_chain[:4]  # Extract PDB ID
    chain_id = pdb_id_chain[4]  # Extract chain ID

    # Define the file paths
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
    output_pdb = f"./data_ext/{pdb_id_chain}"

    if not os.path.exists(pdb_file):
        print(f"File {pdb_file} not found!")
        return

    # Extract the chain and save the PDB and FASTA files
    extract_chain(pdb_file, chain_id, output_pdb)

if __name__ == "__main__":
    main()
