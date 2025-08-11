import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from pathlib import Path


#deep learning hyperparameters

#learning rate for backpropagation:
ETA = 5e-5

#Weight-decay (scaling factor for L2 regulatization to prevent overfit)

LAMBDA = 1e-7

#configure file paths to load data and save models
home_directory = Path(__file__).parent.parent
model_directory = home_directory / "models"

RVAE_encoder_path = model_directory / "RVAE_encoder.pth"
RVAE_deconder_path = model_directory / "RVAE_decoder.pth"
RVAE_regressor_path = model_directory / "RVAE_regressor.pth"

