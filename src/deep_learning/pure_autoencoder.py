import utils
from regression_autoencoder_model import Regression_Autoencoder, model_directory, recon_loss, regression_loss, RAE_loss
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


#configure file paths to save models
print(f"model directory: {model_directory}")
pure_encoder_path = model_directory / "pure_encoder.pth"
pure_decoder_path = model_directory / "pure_decoder.pth"

#deep learning hyperperameters

#learning rate for backpropagation
ETA = 1e-5

#Weight-decay is scaling factor for L2 regularization
LAMBDA = 1e-9



EPOCHS = 70

def train_model(load_old_model = True):
    #load data
    train_loader , val_loader, test_loader = utils.dataloader(utils.preprocess())
    
    #get shape of inputs by iterating once through a DataLoader
    for X_batch, _ in train_loader:
        print("Input batch shape", X_batch.shape)
        input_size = X_batch.shape[1]
        break

    #use CUDA to make use of any avilable NVIDIA GPus, if not just use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Regression_Autoencoder(input_size)
    if load_old_model:
        encoder_state = torch.load(pure_encoder_path, map_location = device)
        decoder_state = torch.load(pure_decoder_path, map_location = device)
        model.encoder.load_state_dict(encoder_state)
        model.decoder.load_state_dict(decoder_state)
        print("Old model loaded")

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA, weight_decay=LAMBDA)

    #epoch losses stores the normalised training loss of each epoch
    epoch_losses=[]
    #training_losses = []
    validation_losses = []
    
    model.to(device)

    print("Initializing training on", device)
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for X_batch,_ in train_loader:
            
            reconstructed = model.forward(X_batch)
            
            loss = loss_function(reconstructed, X_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            #training_losses.append(loss.item())
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)
        #calculate validation loss for epoch
        with torch.no_grad():
            val_loss = 0
            for X_batch, _ in val_loader:
                reconstructed_val = model.forward(X_batch)
                val_loss += loss_function(reconstructed_val, X_batch).item()
            val_loss /= len(val_loader)
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
        for X_batch,_ in test_loader:
            reconstructed_test = model(X_batch)
            test_loss += loss_function(reconstructed_test, X_batch).item()
        test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")

    #Save model parameters for later use, separating encoder and decoder
    torch.save(model.encoder.state_dict(), pure_encoder_path)
    torch.save(model.decoder.state_dict(), pure_decoder_path)
    
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
    #save plot
    plt.savefig("./autoencoder_losses.png")

    plt.show(block=block)

    



if __name__ == "__main__":
    train_model(load_old_model = True) 