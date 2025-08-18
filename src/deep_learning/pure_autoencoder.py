import dl_utils
from regression_autoencoder_model import Regression_Autoencoder, model_directory, recon_loss, regression_loss, RAE_loss
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


#configure file paths to save models

pure_encoder_path = model_directory / "pure_encoder.pth"
pure_decoder_path = model_directory / "pure_decoder.pth"


#deep learning hyperperameters

#learning rate for backpropagation
ETA = 5e-5

#Weight-decay is scaling factor for L2 regularization
LAMBDA = 1e-7



EPOCHS = 70

def main():
    #load data
    train_loader , val_loader, test_loader = dl_utils.dataloader(dl_utils.preprocess())
    
    #get shape of inputs by iterating once through a DataLoader
    for X_batch, _ in train_loader:
        print("Input batch shapeL", X_batch.shape)
        input_size = X_batch.shape[1]
        break

    model = Regression_Autoencoder(input_size)

    loss_function = recon_loss()
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
    plt.show(block=block)

    #save plot
    plt.savefig("./autoencoder_losses.png")
    plt.close('all')



if __name__ == "__main__":
    main()