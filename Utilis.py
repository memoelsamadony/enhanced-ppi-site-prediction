import os
import numpy as np


#Add path here accoding to your device
data_path='./train/' #output folder path for features
Software_path = "./"    
dssp = Software_path + "dssp-3.1.4/mkdssp" #dssp software folder path for features extraction
PSIBLAST = Software_path + "Blastx/bin/psiblast" #PSIBLAST software folder path for features extraction
HHBLITS = Software_path + "hh-suite/build/bin/hhblits" #HH-SUITE software folder path for features extraction
UR90 = "./unirefdb/uniref90.fasta" #database for pssm path #PSIBLAST DATABASE folder path for features extraction
HHDB = "./hmmDB/uniclust30_2017_10" #database hmm path #HH-SUITE DATABASE folder path for features extraction


AA_SYM = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
          "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
AA_AABR = [index for index in "ACDEFGHIKLMNPQRSTVWY"]
aa_dict = dict(zip(AA_SYM, AA_AABR))

Max_pssm = np.array([8, 9, 9, 9, 12, 9, 8, 8, 12, 9, 7, 9, 11, 10, 9, 8, 8, 13, 10, 8])
Min_pssm = np.array([-11,-12,-13,-13,-12,-13,-13,-12,-13,-13,-13,-13,-12,-12,-13,-12,-12,-13,-13,-12])
Max_hhm = np.array([10655,12141,12162,11354,11802,11835,11457,11686,11806,11262,11571,11979,12234,11884,11732,11508,11207,11388,12201,11743])
Min_hhm = np.zeros(20)



error_code_dic = {"PDB not exist": 1, "chain not exist": 2,  "DSSP too long": 3, "Fail to pad DSSP": 4,"software path error":5, "database path error" : 6}
    
def process_pssm(pssm_file):
    with open(pssm_file, "r") as f:
        lines = f.readlines()
    pssm_feature = []
    for line in lines:
        if line == "\n":
            continue
        record = line.strip().split()
        if record[0].isdigit():
            pssm_feature.append([int(x) for x in record[2:22]])
    pssm_feature = (np.array(pssm_feature) - Min_pssm) / (Max_pssm - Min_pssm)

    return pssm_feature



def process_hhm(hhm_file):
    with open(hhm_file, "r") as f:
        lines = f.readlines()
    hhm_feature = []
    p = 0
    while lines[p][0] != "#":
        p += 1
    p += 5
    for i in range(p, len(lines), 3):
        if lines[i] == "//\n":
            continue
        feature = []
        record = lines[i].strip().split()[2:-1]
        for x in record:
            if x == "*":
                feature.append(9999)
            else:
                feature.append(int(x))
        hhm_feature.append(feature)
    hhm_feature = (np.array(hhm_feature) - Min_hhm) / (Max_hhm - Min_hhm)

    return hhm_feature




def prepare_hhm(Input_ID):
    if os.path.exists(data_path + "hhm/{}.hhm".format(Input_ID)) == False:
        os.system("{0} -i {2}{1}.fa -ohhm {2}hhm/{1}.hhm -oa3m {2}{1}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {2}{1}.hhr".format(HHBLITS, Input_ID, data_path, HHDB))
    
    
    hhm_matrix = process_hhm(data_path + "hhm/" + Input_ID + ".hhm")
    
    np.save(data_path + "hhm/" + Input_ID, hhm_matrix)
    





def prepare_pssm(Input_ID):
    if os.path.exists(data_path + "pssm/{}.pssm".format(Input_ID)) == False:
        os.system("{0} -db {1} -num_iterations 3 -num_alignments 1 -num_threads 2 -query {3}{2}.fa -out {3}{2}.bla -out_ascii_pssm {3}pssm/{2}.pssm".format(PSIBLAST, UR90, Input_ID, data_path))


    pssm_matrix = process_pssm(data_path + "pssm/" + Input_ID + ".pssm")
    np.save(data_path + "pssm/" + Input_ID, pssm_matrix)

