#!/usr/bin/env python
# coding: utf-8
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, matthews_corrcoef
# from google.colab import drive
import os
import torch
import torch.utils.data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)