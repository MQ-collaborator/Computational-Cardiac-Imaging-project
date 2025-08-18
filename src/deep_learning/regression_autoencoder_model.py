from torch import nn
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
home_directory = Path(__file__).parent.parent.parent
model_directory = home_directory / "models"




#implement a Regression loss variational autoencoder
# Make sure you don't accidentally put in additional processing 'layers' (problem in earlier versions)
class Regression_Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,latent_dim) #2 layers of same size as latent dimension (consecutive). no activatino to give raw latent space

        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,input_size) # no activation to match normalization of inputs
            #INPUTS MUST come in as 0-1 values so that we can predict them with a ReLU
        )

        self.linear_regressor = nn.Sequential(
            nn.Linear(latent_dim,1)
        )

        self.nn_prediction = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,1),

        )
    def encode(self,x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self,z):
        decoded = self.decoder(z)
        return decoded
    
    def forward(self,x):
        return self.decode(self.encoder(x))
    
    def linear_regress(self,z):
        prediction = self.linear_regressor(z)
        return prediction
    
    def nn_predict(self,z):
        prediction = self.nn_prediction(z),
        return prediction

def recon_loss(x,x_hat):
    recon_loss = nn.MSELoss(x,x_hat)

    return recon_loss

def regression_loss(y,y_pred):
    #pretty simple to use L2 loss for regression
    return nn.MSELoss(y,y_pred)

def RAE_loss(x, x_hat, y_pred, y, beta = 0.2):
    #mixed loss function for reconstruction and rergession
    recon_loss = recon_loss(x, x_hat)
    regression_loss = regression_loss(y, y_pred)

    hybrid_loss = regression_loss + beta * recon_loss
    return hybrid_loss