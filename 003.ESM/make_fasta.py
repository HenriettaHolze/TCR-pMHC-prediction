
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
    print(len(mhc_sequences[0]), len(pep_sequences[0]), len(tcr_sequences[0]), dataset_X.shape)
    df_sequences = pd.DataFrame(
        {"MHC": mhc_sequences, "peptide": pep_sequences, "tcr": tcr_sequences}
    )
    return df_sequences


data_list = []
target_list = []


for fp in glob.glob("../hackathon_data_scripts/data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]

    data_list.append(data)
    target_list.append(targets)

X_train = np.concatenate(data_list[:-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples, nx, ny)

X_val = np.concatenate(data_list[-1:])
y_val = np.concatenate(target_list[-1:])
nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples, nx, ny)

name_list = ['X_val', 'X_train']
for j, data in enumerate([X_val, X_train]):
    complex_sequences = extract_sequences(data)

    # MHCs
    MHC_list = np.array(complex_sequences["MHC"], dtype=str)
    unique_mhc = np.unique(MHC_list)
   
    MHC_records = []
    for i, MHC in enumerate(unique_mhc):
        seq = SeqRecord(Seq(str(MHC)), id=f'MHC-{i}',description='')
        MHC_records.append(seq)
    
    SeqIO.write(MHC_records, f'sequences/MHC-{name_list[j]}.fasta', 'fasta')

    # Peptides
    peptide_list = np.array(complex_sequences["peptide"], dtype=str)
    unique_peptide = np.unique(peptide_list)
   
    peptide_records = []
    for i, peptide in enumerate(unique_peptide):
        seq = SeqRecord(Seq(str(peptide)), id=f'peptide-{i}',description='')
        peptide_records.append(seq)
    
    SeqIO.write(peptide_records, f'sequences/peptide-{name_list[j]}.fasta', 'fasta')

    # TCRs
    tcr_list = np.array(complex_sequences["tcr"], dtype=str)
    unique_tcr = np.unique(tcr_list)
   
    tcr_records = []
    for i, tcr in enumerate(unique_tcr):
        seq = SeqRecord(Seq(str(tcr)), id=f'tcr-{i}',description='')
        tcr_records.append(seq)
    
    SeqIO.write(tcr_records, f'sequences/tcr-{name_list[j]}.fasta', 'fasta')