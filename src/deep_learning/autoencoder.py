import dl_utils
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

#deep learning hyperperameters

#learning rate for backpropagation
ETA = 5e-5

#Weight-decay is scaling factor for L2 regularization
LAMBDA = 1e-7

#file path for saving model parameters within current deep_learning directory
encoder_path = 'encoder.pth'
decoder_path = 'decoder.pth'

EPOCHS = 70
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(32,16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.Sigmoid(),
            nn.Linear(32,input_size)
        )
    
    def forward(self,x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
def main():
    #load data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dl_utils.split_and_normalize(dl_utils.preprocess(), mode=1)
    print("Shape of X_train:", X_train.shape)
    print("Shape of Y_train:", Y_train.shape)
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    X_val = torch.from_numpy(X_val).float()
    Y_val = torch.from_numpy(Y_val).float()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()


    input_size = X_train.shape[1]
    
    model = AutoEncoder(input_size)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA, weight_decay=LAMBDA)

    #epoch losses stores the normalised training loss of each epoch
    epoch_losses=[]
    #training_losses = []
    validation_losses = []
    #use CUDA to make use of any avilable NVIDIA GPus, if not just use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Initializing training on", device)
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for patient in X_train:
            reconstructed = model(patient)
            
            loss = loss_function(reconstructed, patient)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            #training_losses.append(loss.item())
            epoch_loss += loss.item()
        epoch_loss /= len(X_train)
        epoch_losses.append(epoch_loss)
        #calculate validation loss for epoch
        with torch.no_grad():
            val_loss = 0
            for val_patient in X_val:
                reconstructed_val = model(val_patient)
                val_loss += loss_function(reconstructed_val, val_patient).item()
            val_loss /= len(X_val)
            validation_losses.append(val_loss)
        #stop training if validation loss starts to increase for multiple epochs
        if epoch % 4 == 0 or epoch == EPOCHS - 1:
            #if average of the last 2 validations losses is greater than the average of the previous 2, stop training
            if len(validation_losses) > 3 and sum(validation_losses[-2:]) > sum(validation_losses[-4:-2]):
                print("Validation loss increased, stopping training early.")
                break
        print(f"Epoch {epoch+1} / {EPOCHS}, Epoch Loss: {epoch_losses[-1]:.6f}, Validation Loss: {val_loss:.6f}")

    #test the model on the test set
    test_loss = 0
    with torch.no_grad():
        for test_patient in X_test:
            reconstructed_test = model(test_patient)
            test_loss += loss_function(reconstructed_test, test_patient).item()
        test_loss /= len(X_test)
    print(f"Test Loss: {test_loss:.6f}")

    #Save model parameters for later use, separating encoder and decoder
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    
    plot_data(epoch_losses, validation_losses, block=True)


def plot_data(loss1, loss2, block=False):
    #close all currently open figures
    plt.close('all')
    plt.plot(loss1, label='Training Loss')
    #reduce line thickness for better visibility
    plt.gca().lines[-1].set_linewidth(3)

    #increase line thickness of validation loss and include datapoints
    plt.plot(loss2, label='Validation Loss')
    plt.gca().lines[-1].set_linewidth(3)
    plt.scatter(range(len(loss2)), loss2, color='orange', s=10,)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show(block=block)

    #save plot
    plt.savefig("./autoencoder_losses.png")
    plt.close('all')



if __name__ == "__main__":
    main()