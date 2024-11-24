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

# Define the plotting function
def plot_generated_reconstructions(reconstructions, labels, folder, idx, classes):
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
        ax.scatter(rec[0].cpu().detach().numpy(), rec[1].cpu().detach().numpy(), label=classes[labels[i]])
        ax.set_title(f'{classes[labels[i]]}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()

    # Save the plot
    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.savefig(os.path.join(folder, f'reconstruction_{idx}_labels_0_to_5.png'))
    plt.close()

if __name__ == "__main__":
    # Define the classes
    classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]

    # Load models
    VQVAE = torch.load(VQVAE_PATH).to(device)
    VQVAE.eval()  
    modelTransformer = torch.load(MONAI_TRANSFORMER_MODEL_PATH).to(device)
    modelTransformer.eval()
    modelGPT2 = torch.load(TRANSFORMER_MODEL_PATH).to(device)
    modelGPT2.eval()

    # Load data
    train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
        classes=classes,
        iq_samples=iq_samples,
        samples_per_class=samples_per_class,
        batch_size=4
    )
  
    

    # Generate and plot reconstructions for Transformer model
    for j in range(0, 10):
        all_reconstructions = []
        labels = list(range(len(classes)))  # Dynamically adjust based on the number of classes
        for i in labels:
            new_indices = torch.tensor(modelTransformer.generate(torch.tensor([[i]]).to(device), max_new_tokens=513), device=device)
            reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:, 1:]))
            all_reconstructions.append(reconstruction.squeeze()) 

        # Stack all reconstructions
        all_reconstructions = torch.stack(all_reconstructions)
        #print(all_reconstructions.shape)
        plot_generated_reconstructions(all_reconstructions, labels, eval_folder, j, classes)

    # Changing evaluation folder
    eval_folder = f'EvaluationResults/GPT2_results/'

    # Generate and plot reconstructions for GPT2 model
    for j in range(0, 10):
        all_reconstructions = []
        labels = list(range(len(classes)))
        for i in labels:
            new_indices = torch.tensor(modelGPT2.generate(torch.tensor([[i]]).to(device), max_new_tokens=513), device=device)
            reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:, 1:]))
            all_reconstructions.append(reconstruction.squeeze()) 

        # Stack all reconstructions
        all_reconstructions = torch.stack(all_reconstructions)
        #print(all_reconstructions.shape)
        plot_generated_reconstructions(all_reconstructions, labels, eval_folder, j, classes)
