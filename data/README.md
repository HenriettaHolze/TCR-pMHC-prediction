## Description of data

The data found in this folder has been augmented by the embeddings from the esm transformer.

The files can be loaded with np.load, as done in the .ipynb scripts.

When loaded as a numpy array it will have 3 dimensions

1. Sequences (1 row = 1 sequence)

2. Length : a row here corresponds to a specific residue

3. features
    * 0 : 20 can be used to get one hot encoding for that residue
    * 21 : 54 contains energy terms as described in the masters
    * 54 : 1334 contains esm representations
        * For the mean_emb : Only the first 3 rows contain the global terms for the mhc ([0]), peptide ([1]) and tcr ([2]), which can be used in a dense network
        * For the emb (not uploaded yet) : The representations have been aligned with the residues from the peptide, such that residues have same position as the 1280 vector representing that position
        This can probably be used in a CNN/LSTM as done with the sequences


## PCA Data

1. Sequences (1 row = 1 sequence)

2. Length : a row here corresponds to a specific residue (no MHC molecule, so it starts from position 179 from the original dataset)

3. features
    * 0 : 20 can be used to get one hot encoding for that residue
    * 21 : 54 contains energy terms as described in the masters
    * 54 : 104 contains first 50 PC of esm representations transformed with PCA 
        * For the mean_emb : Only the first 3 rows contain the global terms for the mhc ([0]), peptide ([1]) and tcr ([2]), which can be used in a dense network
        * For the emb (not uploaded yet) : The representations have been aligned with the residues from the peptide, such that residues have same position as the 1280 vector representing that position
        This can probably be used in a CNN/LSTM as done with the sequences
