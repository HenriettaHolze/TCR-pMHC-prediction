import argparse
import glob, os, tempfile, zipfile

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score

# Needed to predict
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--input-zip')
args = parser.parse_args()
print(args)

# Load files
filenames = []
dfs = []
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(args.input_zip) as zip:
        files = [file for file in zip.infolist() if file.filename.endswith(".npz")]
        for file in files:
            zip.extract(file, tmpdir)
            
        # Load npz files
        data_list = []

        for fp in glob.glob(tmpdir + "/*input.npz"):
            data = np.load(fp)["arr_0"]

            data_list.append(data)
         
X_test = np.concatenate(data_list[:])
nsamples, nx, ny = X_test.shape
print("test set shape:", nsamples,nx,ny)

test_ds = []
for i in range(len(X_test)):
    test_ds.append([np.transpose(X_test[i])])

bat_size = 64
print("\nNOTE:\nSetting batch-size to", bat_size)
test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=bat_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")

def predict(net, test_ldr):
    net.eval()
    test_preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_ldr): ###
            x_batch_val = data[0].float().detach()

            output = net(x_batch_val)
            preds = np.round(output.detach())
            test_preds += list(preds.data.numpy().flatten()) 
        
    return(test_preds)

    

# import trained model
model = Net(num_classes = 1)
model.load_state_dict(torch.load("src/model.pt"))
model.eval()

y_pred = predict(model, test_ldr)

# Write y_true, y_pred to disk
outname = "predictions.csv"
print("\nSaving TEST set y_pred to", outname)
df_performance = pd.DataFrame({"ix": range(len(y_pred)), "prediction": y_pred},)
df_performance.to_csv(outname, index=False)

print(open(outname, 'r').read())
