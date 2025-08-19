from dl_utils import home_directory
from regression_autoencoder_model import Regression_Autoencoder, model_directory, recon_loss, regression_loss, RAE_loss
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from encode_space import load_embeddings


latent_regression_coefficients_path = home_directory / "models" / "latent_regressor.pth"



#deep learning hyperperameters

#learning rate for backpropagation
ETA = 3e-5

#Weight-decay is scaling factor for L2 regularization
LAMBDA = 1e-9



EPOCHS = 4

def train_model(load_old_model = False):
    #load embeddings and labels
    train_embeddings, val_embeddings, test_embeddings, train_labels, val_labels, test_labels = load_embeddings()


    #use CUDA to make use of any avilable NVIDIA GPus, if not just use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Regression_Autoencoder()
    if load_old_model:
        regressor_state = torch.load(latent_regression_coefficients_path, map_location = device)
        model.linear_regressor.load_state_dict(regressor_state)

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
        for x,y in zip(train_embeddings, train_labels):
            
            prediction = model.linear_regress(x)
            
            loss = loss_function(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            #training_losses.append(loss.item())
            epoch_loss += loss.item()

        epoch_loss /= len(train_embeddings)
        epoch_losses.append(epoch_loss)
        #calculate validation loss for epoch
        with torch.no_grad():
            val_loss = 0
            for x,y in zip(val_embeddings, val_labels):
                prediction = model.linear_regress(x)

                val_loss += loss_function(prediction, y).item()
            val_loss /= len(val_embeddings)
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
        for x,y in zip(test_embeddings, test_labels):
            prediction = model.linear_regress(x)
            test_loss += loss_function(prediction,y).item()
        test_loss /= len(test_embeddings)
    print(f"Test Loss: {test_loss:.6f}")

    #Save model parameters for later use, separating encoder and decoder
    torch.save(model.linear_regressor.state_dict(), latent_regression_coefficients_path)

    #create predictions for all data points so we can evalualte effect of phenotype on each
    #save predictions into numpy arrays
    train_regression_predictions = np.zeroes(train_labels.shape())

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
    train_model(load_old_model = False)