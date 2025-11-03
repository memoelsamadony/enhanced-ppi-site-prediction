import os
import gzip

# Mapping for converting three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "ASX": "B", "GLX": "Z", "XAA": "X",
    "UNK": "X"
}

def convert_chain_id(chain_id):
    """Convert numeric chain IDs to alphabetic ones (1 -> A, 2 -> B, etc.)."""
    if chain_id.isdigit():
        return chr(ord('A') + int(chain_id) - 1)
    return chain_id

def extract_chain_from_pdb(pdb_file, chain_id, output_pdb, output_fasta):
    # Handle compressed PDB files
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as f:
            pdb_lines = f.readlines()
    else:
        with open(pdb_file, 'r') as f:
            pdb_lines = f.readlines()

    output_lines = []
    sequence = []
    found_chain = False
    original_chain_id = chain_id  # Keep track of the original chain ID

    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            current_chain_id = line[21].strip()
            
            # Convert numeric chain ID to alphabet if necessary
            new_chain_id = convert_chain_id(current_chain_id)
            
            if current_chain_id == original_chain_id or new_chain_id == chain_id:
                found_chain = True
                # Replace the chain ID in the line
                line = line[:21] + new_chain_id + line[22:]
                output_lines.append(line)
                
                # Extract residue information
                resname = line[17:20].strip()  # Extract 3-letter residue code
                if resname in THREE_TO_ONE:
                    sequence.append(THREE_TO_ONE[resname])
                else:
                    sequence.append('X')  # Unknown or non-standard residues

    if not found_chain:
        print(f"Chain {chain_id} not found in {pdb_file}")
        return

    # Write the modified PDB file
    with open(output_pdb, 'w') as pdb_output:
        pdb_output.writelines(output_lines)
    print(f"Saved modified chain {chain_id} to {output_pdb}")

    

def main():
    pdb_dir = "./data_ext/"  # Modify this to your PDB directory
    pdb_id_chain = "7NEGB"  # Example PDB ID and chain (e.g., 3mcbA)

    pdb_id = pdb_id_chain[:4]  # Extract PDB ID
    chain_id = pdb_id_chain[4]  # Extract chain ID

    # Define the file paths
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
    output_pdb = f"./data_ext/{pdb_id_chain}"
    output_fasta = f"./data_ext/{pdb_id_chain}.fa"

    if not os.path.exists(pdb_file):
        print(f"File {pdb_file} not found!")
        return

    # Extract the chain and save the PDB and FASTA files
    extract_chain_from_pdb(pdb_file, chain_id, output_pdb, output_fasta)

if __name__ == "__main__":
    main()
