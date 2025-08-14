import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from pathlib import Path


#deep learning hyperparameters

#learning rate for backpropagation:
ETA = 5e-5

#Weight-decay (scaling factor for L2 regulatization to prevent overfit)

LAMBDA = 1e-7

#set number of training epochs. Use 1 for testing
EPOCHS =1 

#configure file paths to load data and save models
home_directory = Path(__file__).parent.parent
model_directory = home_directory / "models"

RVAE_encoder_path = model_directory / "RVAE_encoder.pth"
RVAE_deconder_path = model_directory / "RVAE_decoder.pth"
RVAE_regressor_path = model_directory / "RVAE_regressor.pth"

#implement a Regression loss variational autoencoder
# Make sure you don't accidentally put in additional processing 'layers' (problem in earlier versions)
class Autoencoder(nn,Module):
    def __init__(self, input_size)
        self.encoder = nn.Sequential)