import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, matthews_corrcoef
import random


def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
# Set seed
setup_seed(1)

X_train = np.load('../data/X_train_pca_100.npz')['arr_0']

X_val = np.load('../data/X_val_pca_100.npz')['arr_0']

y_train = np.load('../data/y_train.npz')['arr_0']
y_val = np.load('../data/y_val.npz')['arr_0']

nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples, nx, ny)

p_neg = len(y_train[y_train == 1]) / len(y_train) * 100
print("Percent positive samples in train:", p_neg)

p_pos = len(y_val[y_val == 1]) / len(y_val) * 100
print("Percent positive samples in val:", p_pos)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([np.transpose(X_train[i]), y_train[i]])

val_ds = []
for i in range(len(X_val)):
    val_ds.append([np.transpose(X_val[i]), y_val[i]])
    
bat_size = 64
print("\nNOTE:\nSetting batch-size to", bat_size)
train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=bat_size, shuffle=True)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")



########## Define Network ###########
# Hyperparameters
# input_size = 420
_, input_size, n_features = X_train.shape
n_local_feat = 100 ##
n_global_feat = 27 
num_classes = 1
# learning_rate = 0.01
learning_rate = 0.001
weight_decay = 0.0005

loss_weight = sum(y_train) / len(y_train)
print("loss weight", loss_weight)

class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()   
        self.bn0 = nn.BatchNorm1d(n_local_feat)
        self.conv1 = nn.Conv1d(in_channels=n_local_feat, out_channels=300, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(300)
        
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)
        
        ######## code from master thesis 
        
        self.rnn1 = nn.LSTM(input_size=100,hidden_size=26,num_layers=1, batch_first=True, bidirectional = True)
        self.rnn2 = nn.LSTM(input_size=26*2,hidden_size=26,num_layers=1, batch_first=True, bidirectional = True)
        self.bn1 = nn.BatchNorm1d(26*2 + n_global_feat)
        self.drop = nn.Dropout(p = 0.6) # Dunno if dropout should be even higher?? - Christian
        self.fc1 = nn.Linear(26*2 + n_global_feat, 26*2 + n_global_feat)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        ########
        
        # since we add new features in this step, we have to use batch normalization again
        
        # if we pipe the global terms innto the fc, we should have more than just 1
        self.fc2 = nn.Linear(26*2 + n_global_feat, (26*2 + n_global_feat))
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(26*2 + n_global_feat, num_classes)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # local_features = x[:, 20:27, :] ##
        # global features are the same for the whole sequence -> take first value
        global_features = x[:, 27:54, 0]
        esm_features = x[:, 54:, :]
        #local_features = torch.cat((local_features, esm_features),dim=1 )
        local_features = esm_features ##
        
        ######## code from master thesis
        x = self.bn0(local_features)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn1(x)
        x = self.drop(x)
        x, (h, c) = self.rnn2(x)
        x = self.drop(x)
        x, (h, c) = self.rnn2(x)
        # concatenate bidirectional output of last layer
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        # add global features
        x = torch.cat((cat, global_features), dim=1)
        x = self.drop(x)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc3(x))
        ########
        
        return x
    
######### RUNNING TRAINING ############

net = Net(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss(reduction='none')  # for weighted loss
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer = optim.Adam(net.parameters(), lr=learning_rate,
    weight_decay=weight_decay,
    amsgrad=True
)

num_epochs = 100

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
losses = []
val_losses = []

# for early stopping
no_epoch_improve = 0
min_val_loss = np.Inf

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0

    net.train()
    train_preds, train_preds_auc, train_targs = [], [], []
    for batch_idx, (data, target) in enumerate(train_ldr):
        X_batch = data.float().detach().requires_grad_(True).cuda(device)
        target_batch = torch.tensor(np.array(target), dtype=torch.float).cuda(device).unsqueeze(1)

        optimizer.zero_grad()
        output = net(X_batch)

        # calculate weighted loss
        intermediate_loss = criterion(output, target_batch)
        weights = torch.cuda.FloatTensor(abs(target_batch - loss_weight))
        batch_loss = torch.mean(weights * intermediate_loss)
        # batch_loss = criterion(output, target_batch)
        
        batch_loss.backward()
        optimizer.step()

        preds = np.round(output.detach().cpu())
        preds_auc = output.detach().cpu()
        train_targs += list(np.array(target_batch.cpu()))
        train_preds += list(preds.data.numpy().flatten())
        train_preds_auc += list(preds_auc.data.numpy().flatten())
        cur_loss += batch_loss.detach()

    losses.append(cur_loss.cpu() / len(train_ldr.dataset))

    net.eval()
    ### Evaluate validation
    val_preds, val_preds_auc, val_targs = [], [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_ldr):  ###
            x_batch_val = data.float().detach().cuda(device)
            y_batch_val = target.float().detach().cuda(device).unsqueeze(1)

            output = net(x_batch_val)

            # calculate weighted loss
            intermediate_loss = criterion(output, y_batch_val)
            weights = torch.cuda.FloatTensor(abs(y_batch_val - loss_weight))
            val_batch_loss = torch.mean(weights * intermediate_loss)
            # val_batch_loss = criterion(output, y_batch_val)

            preds = np.round(output.detach().cpu())
            val_preds += list(preds.cpu().data.numpy().flatten())
            preds_auc = output.detach().cpu()
            val_preds_auc += list(preds_auc.cpu().data.numpy().flatten())
            val_targs += list(np.array(y_batch_val.cpu()))
            val_loss += val_batch_loss.cpu().detach()

        val_losses.append(val_loss.cpu() / len(val_ldr.dataset))
        print("\nEpoch:", epoch + 1)

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)


        print(
            "Training loss:",
            losses[-1].item(),
            "Validation loss:",
            val_losses[-1].item(),
            end="\n",
        )
        print(
            "MCC Train:",
            matthews_corrcoef(train_targs, train_preds),
            "MCC val:",
            matthews_corrcoef(val_targs, val_preds),
        )
        
    # Early stopping: no improvement in validation loss in 10 consecutive epochs
    if (val_loss / len(X_val)).item() < min_val_loss:
        no_epoch_improve = 0
        min_val_loss = (val_loss / len(X_val))
        torch.save(net, 'early_stopping_model.pt')
        best_epoch = epoch + 1
    else:
        no_epoch_improve +=1
    if no_epoch_improve == 10:
        print("Early stopping\n")
        break


############ PERFORMANCE EVALUATION ################

# Plots of training epochs
epoch = np.arange(1, len(train_acc) + 1)
plt.figure()
plt.plot(epoch, losses, "r", epoch, val_losses, "b")
plt.legend(["Train Loss", "Validation Loss"])
plt.vlines(best_epoch, ymin=0, ymax=0.005, linestyles='dashed')
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.savefig('../plots/loss_curve')

epoch = np.arange(1, len(train_acc) + 1)
plt.figure()
plt.plot(epoch, train_acc, "r", epoch, valid_acc, "b")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.vlines(best_epoch, ymin=0, ymax=0.9, linestyles='dashed')
plt.xlabel("Epoch"), plt.ylabel("Acc")
plt.savefig('../plots/accuracy_curve')




# Performance evaluation metrics of final model

print('\n\n\nFinal Model Performance:')
final_model = torch.load('early_stopping_model.pt')
final_model.train()
train_preds, train_preds_auc, train_targs = [], [], []
for batch_idx, (data, target) in enumerate(train_ldr):
    X_batch = data.float().detach().requires_grad_(True).cuda(device)
    target_batch = torch.tensor(np.array(target), dtype=torch.float).cuda(device).unsqueeze(1)
    
    output = final_model(X_batch)
    preds = np.round(output.detach().cpu())
    preds_auc = output.detach().cpu()
    train_targs += list(np.array(target_batch.cpu()))
    train_preds += list(preds.data.numpy().flatten())
    train_preds_auc += list(preds_auc.data.numpy().flatten())

final_model.eval()
val_preds, val_preds_auc, val_targs = [], [], []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_ldr):  ###
        x_batch_val = data.float().detach().cuda(device)
        y_batch_val = target.float().detach().cuda(device).unsqueeze(1)

        output = final_model(x_batch_val)

        preds = np.round(output.cpu().detach())
        val_preds += list(preds.cpu().data.numpy().flatten())
        preds_auc = output.cpu().detach()
        val_preds_auc += list(preds_auc.data.numpy().flatten())
        val_targs += list(np.array(y_batch_val.cpu()))
        val_loss += val_batch_loss.cpu().detach()

print("MCC Train:", matthews_corrcoef(train_targs, train_preds))
print("MCC Test:", matthews_corrcoef(val_targs, val_preds))

prec_val = metrics.precision_score(val_targs, val_preds)
rec_val = metrics.recall_score(val_targs, val_preds)
f1_val = 2 * ((prec_val * rec_val) / (prec_val + rec_val))

print("Precision Test:", prec_val)
print("Recall Test:", rec_val)
print("F1 Test:", f1_val)

print("Confusion matrix train:", confusion_matrix(train_targs, train_preds), sep="\n")
print("Confusion matrix test:", confusion_matrix(val_targs, val_preds), sep="\n")


def plot_roc(targets, predictions):
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # plot ROC
    plt.figure()
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    # plt.show()


plot_roc(train_targs, train_preds_auc)
plt.title("Training AUC")
plt.savefig('../plots/training_AUC')
plot_roc(val_targs, val_preds_auc)
plt.title("Validation AUC")
plt.savefig('../plots/validation_AUC')
plt.show()
