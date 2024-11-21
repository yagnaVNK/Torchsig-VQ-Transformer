import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm  
from src.VQVAE import *
from src.utils import *
from src.Transformer import TransformerModel
from src.TransformerMonai import MonaiDecoderOnlyModel
from src.Dataset import getDataLoader
import matplotlib.pyplot as plot
import numpy as np


def plot_batch_reconstruction(original_batch, reconstruction_batch, idx, folder):
    plt.figure(figsize=(12, 12))
    
    for i in range(4):

        plt.subplot(4, 2, 2 * i + 1)
        plt.plot(original_batch[i])
        plt.title(f'Original Signal {i+1}')
        
        plt.subplot(4, 2, 2 * i + 2)
        plt.plot(reconstruction_batch[i])
        plt.title(f'Reconstructed Signal {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"{folder}/context_len_{idx}.png")
    plt.close()



def plot_generated_reconstructions(reconstructions, labels, folder,idx):
    # Create the eval folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Define the number of subplots based on the number of reconstructions
    n_reconstructions = reconstructions.shape[0]
    cols = 3  # Number of columns in subplot grid
    rows = (n_reconstructions + cols - 1) // cols  # Calculate rows needed
    
    # Create a figure with subplots
    plt.figure(figsize=(18, 6 * rows))  # Adjust the size as needed


    # Plot each reconstruction in a subplot
    for i, rec in enumerate(reconstructions):
        if rec.nelement() == 0:
            continue  # Skip plotting if the tensor is empty
        ax = plt.subplot(rows, cols, i + 1)
        ax.scatter(rec[0].cpu().detach().numpy(), rec[1].cpu().detach().numpy(), label=f'Label {labels[i]}')
        ax.set_title(f'Reconstruction Label {labels[i]}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()

    # Save the plot
    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.savefig(os.path.join(folder, f'reconstruction_{idx}_labels_0_to_5.png'))
    plt.close()



if __name__ == "__main__":
    
    VQVAE = torch.load(VQVAE_PATH).to(device)
    VQVAE.eval()  
    modelTransformer = torch.load(MONAI_TRANSFORMER_MODEL_PATH).to(device)
    modelTransformer.eval()
    modelGPT2 = torch.load(TRANSFORMER_MODEL_PATH).to(device)
    modelGPT2.eval()
    
    train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
        classes = classes,
        iq_samples = iq_samples,
        samples_per_class= samples_per_class,
        batch_size=4
    )
  
    for j in range(0,10):
        all_reconstructions = []
        labels = list(range(6))
        for i in labels:
            new_indices = torch.tensor(modelTransformer.generate(torch.tensor([[i]]), max_new_tokens=513), device=device)
            #print(new_indices[:,1:].shape)
            reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:,1:]))
            #print(reconstruction.shape)
            all_reconstructions.append(reconstruction.squeeze()) 

        # Stack all reconstructions into a tensor of shape [6, 2, 1024]
        all_reconstructions = torch.stack(all_reconstructions)
        print(all_reconstructions.shape)
        plot_generated_reconstructions(all_reconstructions, labels, eval_folder,j)

    eval_folder = f'EvaluationResults/GPT2_results/'

    for j in range(0,10):
        all_reconstructions = []
        labels = list(range(6))
        for i in labels:
            new_indices = torch.tensor(modelGPT2.generate(torch.tensor([[i]]), max_new_tokens=513), device=device)
            #print(new_indices[:,1:].shape)
            reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:,1:]))
            #print(reconstruction.shape)
            all_reconstructions.append(reconstruction.squeeze()) 

        # Stack all reconstructions into a tensor of shape [6, 2, 1024]
        all_reconstructions = torch.stack(all_reconstructions)
        print(all_reconstructions.shape)
        plot_generated_reconstructions(all_reconstructions, labels, eval_folder,j)

