#encode latent space and save as tensor. Include labels separately
#important to use model.eval() to disable dropout!!!
import dl_utils
from pure_autoencoder import pure_encoder_path, pure_decoder_path
from regression_autoencoder_model import Regression_Autoencoder
import torch
import numpy as np

def encode_space():
    #load unencoded data
    train_loader , val_loader, test_loader = dl_utils.dataloader(dl_utils.preprocess())

    #get shape of inputs by iterating once through a DataLoader
    for X_batch, _ in train_loader:
        print("Input batch shape", X_batch.shape)
        input_size = X_batch.shape[1]
        break

        #use CUDA to make use of any avilable NVIDIA GPus, if not just use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    #load encoder model
    model = Regression_Autoencoder(input_size)
    encoder_state = torch.load(pure_encoder_path, map_location = device)
    decoder_state = torch.load(pure_decoder_path, map_location = device)
    model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(decoder_state)

    
    # Extract and save
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)
    val_embeddings, val_labels = extract_embeddings(val_loader, model, device)
    test_embeddings, test_labels = extract_embeddings(test_loader, model, device)

    np.savez("latent_embeddings.npz",
            train_embeddings=train_embeddings, train_labels=train_labels,
            val_embeddings=val_embeddings, val_labels=val_labels,
            test_embeddings=test_embeddings, test_labels=test_labels)

    print("Latent embeddings and labels saved to latent_embeddings.npz.")


        

# Function to extract latent embeddings and labels
def extract_embeddings(loader, model, device):
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.encoder(x)
            embeddings.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(labels)



if __name__ == "__main__":
    encode_space()