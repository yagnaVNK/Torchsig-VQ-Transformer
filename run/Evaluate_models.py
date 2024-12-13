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

from top_pr import compute_top_pr as TopPR


def compute_metrics(real_dataset_path, fake_dataset_path):
    """
    Compute metrics using Top PR framework for real and fake datasets.

    Parameters:
    - real_dataset_path: Path to the NumPy file containing the real dataset.
    - fake_dataset_path: Path to the NumPy file containing the fake dataset.

    Returns:
    - A dictionary containing fidelity, diversity, and Top_F1 metrics.
    """
    # Load the datasets
    real_data = np.load(real_dataset_path)
    fake_data = np.load(fake_dataset_path)

    # Ensure both datasets are flattened in the signal dimension
    real_data = real_data.reshape(real_data.shape[0], -1)
    fake_data = fake_data.reshape(fake_data.shape[0], -1)

    # Compute metrics using TopPR
    Top_PR = TopPR(
        real_features=real_data,
        fake_features=fake_data,
        alpha=0.1,  # Weight for fidelity and diversity tradeoff
        kernel="cosine",  # Kernel type
        random_proj=True,  # Whether to use random projection
        f1_score=True  # Whether to compute Top_F1
    )

    # Extract the required metrics
    fidelity = Top_PR.get("fidelity")
    diversity = Top_PR.get("diversity")
    top_f1 = Top_PR.get("Top_F1")

    # Print and return the metrics
    print(f"Fidelity: {fidelity}, Diversity: {diversity}, Top_F1: {top_f1}")
    return {"fidelity": fidelity, "diversity": diversity, "Top_F1": top_f1}

def save_torch_dataset_as_numpy(dataset, output_file):
    """
    Save a PyTorch Dataset as a NumPy ndarray.

    Parameters:
    - dataset: A PyTorch Dataset object with tensors to be saved.
    - output_file: Path to save the dataset as a NumPy file.
    """
    # Initialize a list to hold all samples
    all_samples = []
    print("converting dataset to numpy array")
    # Iterate over the dataset to extract tensors
    for i in tqdm(range(len(dataset))):
        # Assuming each sample in the dataset is a tuple (data, label)
        sample, _ = dataset[i]  # Extract only the data
        all_samples.append(sample.reshape(2048))  # Flatten and convert to numpy array

    # Convert the list of samples to a NumPy array
    numpy_array = np.array(all_samples)
    print(f"Dataset shape: {numpy_array.shape}")

    # Save the NumPy array to a file
    np.save(output_file, numpy_array)
    print(f"Dataset saved to {output_file}")


def generate_fake_dataset(transformer_model, vqvae_model, output_file, num_labels=6, samples_per_label=500, batch_size=128, signal_length=1024):
    """
    Generate a fake dataset using a transformer model and save it as a NumPy array.
    
    Parameters:
    - transformer_model: The trained transformer model for sequence generation.
    - vqvae_model: The VQVAE model for decoding generated sequences.
    - output_file: Path to save the final dataset as a NumPy array.
    - num_labels: Number of labels (classes) to generate samples for.
    - samples_per_label: Number of samples to generate for each label.
    - batch_size: Number of samples to generate in parallel for each label.
    - signal_length: Length of the decoded signal for each sample.
    """
    all_samples = []
    steps_per_label = samples_per_label // batch_size
    fake_indices = []
    for label in range(num_labels):
        print(f"Generating data for label {label}...")
        for _ in tqdm(range(steps_per_label)):
            # Create a batch of the same label
            label_tensor = torch.tensor([[label]] * batch_size).to(device)

            # Generate new indices using the transformer model
            new_indices = transformer_model.generate(
                label_tensor,
                max_new_tokens=signal_length // 2
            )
            fake_indices.append(new_indices[:,1:].cpu().numpy())
            # Decode the generated indices into signals
            reconstructions = vqvae_model.decode(vqvae_model.codebook.lookup(new_indices[:, 1:]))

            # Flatten each reconstructed signal and append it to the dataset
            for rec in reconstructions:
                all_samples.append(rec.cpu().detach().numpy().flatten())
    
    # Convert the list of samples into a NumPy array
    final_dataset = np.array(all_samples)
    print(f"Generated dataset shape: {final_dataset.shape}")
    
    return fake_indices

def plot_codebook_histograms(indices, path, num_classes=6, num_tokens=512):
    """
    Plot histograms of the quantization indices from the VQVAE model.

    Parameters:
    - indices: Numpy array of indices obtained from the quantization.
    - path: Path where the histogram plots will be saved.
    - num_classes: Number of classes for labeling purposes.
    - num_tokens: Number of tokens in the VQVAE's codebook.
    """
    os.makedirs(path, exist_ok=True)
    classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]

    # Assuming indices is a flat list, we need to split it correctly by class
    # This requires knowing the distribution of indices by class - the following is a generic handler
    indices_per_class = np.array_split(indices, num_classes)  # Using array_split to handle uneven splits just in case

    for i, class_indices in enumerate(indices_per_class):
        plt.figure(figsize=(10, 6))
        plt.hist(class_indices, bins=64, color='blue', alpha=0.7)
        plt.title(f'Histogram for Class: {classes[i]}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        outputpath = os.path.join(path, f"Histogram_Class_{classes[i]}.png")
        plt.savefig(outputpath)
        plt.close()



def quantize_dataset(vqvae_model, dataset, batch_size=128):
    """
    Quantize an entire dataset using the VQVAE model and return the indices.

    Parameters:
    - vqvae_model: Pre-trained VQVAE model for quantization.
    - dataset: PyTorch Dataset to be quantized.
    - batch_size: Batch size for processing.

    Returns:
    - A list of quantization indices from the VQVAE model.
    """
    vqvae_model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_indices = []

    for data, _ in tqdm(dataloader):
        data = data.to(device)  # Ensure data is on the same device as model
        with torch.no_grad():
            indices = vqvae_model.codebook.quantize_indices(vqvae_model.encode(data))
        all_indices.extend(indices.cpu().numpy())  # Move indices to CPU and store

    return np.concatenate(all_indices)


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
  
    

    # # Generate and plot reconstructions for Transformer model
    # for j in range(0, 10):
    #     all_reconstructions = []
    #     labels = list(range(len(classes)))  # Dynamically adjust based on the number of classes
    #     for i in labels:
    #         new_indices = torch.tensor(modelTransformer.generate(torch.tensor([[i]]).to(device), max_new_tokens=512), device=device)
    #         reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:, 1:]))
    #         all_reconstructions.append(reconstruction.squeeze()) 

    #     # Stack all reconstructions
    #     all_reconstructions = torch.stack(all_reconstructions)
    #     #print(all_reconstructions.shape)
    #     plot_generated_reconstructions(all_reconstructions, labels, eval_folder, j, classes)

    # # Changing evaluation folder
    # eval_folder = f'EvaluationResults/GPT2_results/'

    # # Generate and plot reconstructions for GPT2 model
    # for j in range(0, 10):
    #     all_reconstructions = []
    #     labels = list(range(len(classes)))
    #     for i in labels:
    #         new_indices = torch.tensor(modelGPT2.generate(torch.tensor([[i]]).to(device), max_new_tokens=512), device=device)
    #         reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices[:, 1:]))
    #         all_reconstructions.append(reconstruction.squeeze()) 

    #     # Stack all reconstructions
    #     all_reconstructions = torch.stack(all_reconstructions)
    #     #print(all_reconstructions.shape)
    #     plot_generated_reconstructions(all_reconstructions, labels, eval_folder, j, classes)


    # real_dataset_path = "src/Saved_datasets/real_dataset.npy"  
    # fake_dataset_path_GPT2 = "src/Saved_datasets/fake_dataset_transformerGPT2.npy" 
    # fake_dataset_path_MONAI = "src/Saved_datasets/fake_dataset_transformerMONAI.npy" 

    # GPT2_Histograms_Path = "EvaluationResults/GPT2_results/CodebookHistograms/" 
    # MONAI_Histograms_Path = "EvaluationResults/Monai_results/CodebookHistograms/" 

    # # Save ds_train as a NumPy array
    # #save_torch_dataset_as_numpy(ds_train, real_dataset_path)

    # fake_indices_GPT2 = generate_fake_dataset(
    #     transformer_model=modelGPT2,
    #     vqvae_model=VQVAE,
    #     output_file=fake_dataset_path_GPT2,
    #     num_labels=6,
    #     samples_per_label=1000,
    #     signal_length=1024
    # )
    # plot_codebook_histograms(fake_indices_GPT2,GPT2_Histograms_Path)
    # metrics = compute_metrics(real_dataset_path, fake_dataset_path_GPT2)
    # print("Computed Metrics GPT2:", metrics)


    # fake_indices_MONAI = generate_fake_dataset(
    #     transformer_model=modelTransformer,
    #     vqvae_model=VQVAE,
    #     output_file=fake_dataset_path_MONAI,
    #     num_labels=6,
    #     samples_per_label=1000,
    #     signal_length=1024
    # )
    # plot_codebook_histograms(fake_indices_MONAI,MONAI_Histograms_Path)
    # metrics = compute_metrics(real_dataset_path, fake_dataset_path_MONAI)
    # print("Computed Metrics MONAI:", metrics)


    indices = quantize_dataset(VQVAE, ds_test)

    # Plot and save histograms
    histogram_path = "EvaluationResults/TestDatasetHistograms/"
    plot_codebook_histograms(indices, histogram_path)
