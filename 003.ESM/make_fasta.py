
import glob
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

def reverseOneHot(encoding):
    """
    Converts one-hot encoded array back to string sequence
    """
    mapping = dict(zip(range(20), "ACDEFGHIKLMNPQRSTVWY"))
    seq = ""
    for i in range(len(encoding)):
        if np.max(encoding[i]) > 0:
            seq += mapping[np.argmax(encoding[i])]
    return seq


def extract_sequences(dataset_X):
    """
    Return DataFrame with MHC, peptide and TCR a/b sequences from
    one-hot encoded complex sequences in dataset X
    """
    mhc_sequences = [reverseOneHot(arr[0:179, 0:20]) for arr in dataset_X]
    pep_sequences = [reverseOneHot(arr[179:188, 0:20]) for arr in dataset_X]
    tcr_sequences = [reverseOneHot(arr[188:, 0:20]) for arr in dataset_X]
    df_sequences = pd.DataFrame(
        {"MHC": mhc_sequences, "peptide": pep_sequences, "tcr": tcr_sequences}
    )
    return df_sequences

path = "hackathon_data_scripts/data/train/"
data_dir = "data/sequences/"
data_list = []
files = ['P3_input.npz', 'P4_input.npz', 'P2_input.npz', 'P1_input.npz']
for fp in files:
    data = np.load(path+fp)["arr_0"]


    data_list.append(data)



X_train = np.concatenate(data_list[:-1])

nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples, nx, ny)

X_val = np.concatenate(data_list[-1:])
X_test = np.load("hackathon_data_scripts/data/final_test/P5_input.npz")['arr_0']
nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples, nx, ny)

name_list = ['X_val', 'X_train', 'X_test']
for j, data in enumerate([X_val, X_train, X_test]):
    complex_sequences = extract_sequences(data)

    # MHCs
    MHC_list = np.array(complex_sequences["MHC"], dtype=str)
    unique_mhc = np.unique(MHC_list)
   
    MHC_records = []
    for i, MHC in enumerate(unique_mhc):
        seq = SeqRecord(Seq(str(MHC)), id=f'MHC-{i}',description='')
        MHC_records.append(seq)

    SeqIO.write(MHC_records, data_dir + f'MHC-{name_list[j]}.fasta', 'fasta')

    # Peptides
    peptide_list = np.array(complex_sequences["peptide"], dtype=str)
    unique_peptide = np.unique(peptide_list)
   
    peptide_records = []
    for i, peptide in enumerate(unique_peptide):
        seq = SeqRecord(Seq(str(peptide)), id=f'peptide-{i}',description='')
        peptide_records.append(seq)

    SeqIO.write(peptide_records, data_dir + f'peptide-{name_list[j]}.fasta', 'fasta')

    # TCRs
    tcr_list = np.array(complex_sequences["tcr"], dtype=str)
    unique_tcr = np.unique(tcr_list)
   
    tcr_records = []
    for i, tcr in enumerate(unique_tcr):
        seq = SeqRecord(Seq(str(tcr)), id=f'tcr-{i}',description='')
        tcr_records.append(seq)

    SeqIO.write(tcr_records, data_dir + f'tcr-{name_list[j]}.fasta', 'fasta')