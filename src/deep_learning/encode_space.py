#encode latent space and save as tensor. Include labels separately
#important to use model.eval() to disable dropout!!!
import dl_utils
from pure_autoencoder import pure_encoder_path, pure_decoder_path
from regression_autoencoder_model import Regression_Autoencoder
import torch
import numpy as np

#store separated embeddings for reuse
helper_directory = r"./helper_data"
latent_embeddings_path = helper_directory + "/embeddings.npz"

def load_embeddings(path = latent_embeddings_path):
    #load latent embeddings from npz file
    all_embeddings = np.load(latent_embeddings_path)
    #separate embeddings into 3 distinct set
    train_embeddings = torch.tensor(all_embeddings["train_embeddings"])
    val_embeddings = torch.tensor(all_embeddings["val_embeddings"])
    test_embeddings = torch.tensor(all_embeddings["test_embeddings"])

    train_labels = torch.tensor(all_embeddings["train_labels"])
    val_labels = torch.tensor(all_embeddings["val_labels"])
    test_labels = torch.tensor(all_embeddings["test_labels"])

    #return all data as pytorch tensors
    return train_embeddings, val_embeddings, test_embeddings, train_labels, val_labels, test_labels

def get_train_phenotypes(variable_name):
    df = dl_utils.preprocess()
    split_indices = np.load("./helper_data/split_indices.npz")
    train_idx = split_indices["train"]

    if variable_name not in df.columns:
        raise ValueError(f"Variable '{variable_name}' not found in phenotype data.")

    return df.iloc[train_idx][variable_name].values

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

    np.savez(latent_embeddings_path,
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