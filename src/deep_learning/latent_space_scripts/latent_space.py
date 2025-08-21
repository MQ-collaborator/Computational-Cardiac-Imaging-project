"""Program to use encoder to convert data to latent space and store values.

These values when saved than then be used to perform PCA or to train a linear regression model

we expect to be able to explain more than 9-% of variance with 16 latent variables"""
import deep_learning.utils as utils
import deep_learning.pure_autoencoder as pure_autoencoder
from deep_learning.pure_autoencoder import AutoEncoder
import pandas as pd
import torch

encoded_data_path = 'encoded_data.csv'

def main():
    #load dataset
    X,Y = utils.split_and_normalize(utils.preprocess(), mode=0)

    #load encoder
    input_size = X.shape[1]
    model = AutoEncoder(input_size)
    model.encoder.load_state_dict(torch.load(pure_autoencoder.encoder_path))

    #encode data
    X_encoded = model.encoder(torch.from_numpy(X).float()).detach().numpy()
    #save encoded data to a csv file
    encoded_df = pd.DataFrame(X_encoded, columns=[f'latent_{i+1}' for i in range(X_encoded.shape[1])])
    #add labels to the encoded data
    encoded_df['SBP_at_MRI'] = Y


    encoded_df.to_csv(encoded_data_path, index=False)
    print(f"Encoded data saved to {encoded_data_path}")
if __name__ == "__main__":
    main()