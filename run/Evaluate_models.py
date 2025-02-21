import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  
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


def generate_fake_dataset(transformer_model,is_conditional, vqvae_model, output_file, num_labels=6, BOS_TOKEN = 64, samples_per_label=64, batch_size=64, signal_length=1024):
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
            if is_conditional:
                # For conditional models, provide both BOS token and label
                
                new_indices = model.generate(torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + label]] * batch_size).to(device), max_new_tokens=512)
                new_indices = new_indices[:,2:]
            else:
                # For unconditional models, only provide BOS token
                new_indices = model.generate(torch.tensor([[BOS_TOKEN]]* batch_size).to(device), max_new_tokens=512)
                new_indices = new_indices[:,1:]

            # Each new indices shape will be (Batch_Size , 512)

            fake_indices.append(new_indices.cpu().numpy()) 
            # Decode the generated indices into signals
            reconstructions = vqvae_model.decode(vqvae_model.codebook.lookup(new_indices))
            
            # Flatten each reconstructed signal and append it to the dataset
            for rec in reconstructions:
                #print(rec.shape)
                all_samples.append(rec.cpu().detach().numpy().flatten())
            print(len(all_samples)) 
            
    # Convert the list of samples into a NumPy array
    final_dataset = np.array(all_samples)
    np.save(output_file,final_dataset)

    print(f"Generated dataset shape: {final_dataset.shape}")
    
    return fake_indices

def plot_codebook_histograms(indices, path, num_classes=6, num_tokens=512):
    """
    Plot histograms of the quantization indices from the VQVAE model.

    Parameters:
    - indices: List of numpy arrays containing the quantization indices
    - path: Path where the histogram plots will be saved
    - num_classes: Number of classes for labeling purposes
    - num_tokens: Number of tokens in the VQVAE's codebook
    """
    os.makedirs(path, exist_ok=True)
    classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]
    
    # Convert indices to numpy array if it isn't already
    indices = np.array(indices)
    
    # Reshape indices to ensure they're properly formatted
    # If indices is a list of arrays, concatenate them first
    if isinstance(indices, list):
        indices = np.concatenate(indices)
    
    # Ensure indices are 2D: (num_samples, sequence_length)
    if indices.ndim > 2:
        indices = indices.reshape(-1, indices.shape[-1])
    
    # Split indices into classes
    samples_per_class = len(indices) // num_classes
    indices_per_class = np.array_split(indices, num_classes)
    
    for i, class_indices in enumerate(indices_per_class):
        plt.figure(figsize=(10, 6))
        # Flatten the indices for plotting
        flat_indices = class_indices.flatten()
        plt.hist(flat_indices, bins=min(64, len(np.unique(flat_indices))), 
                color='blue', alpha=0.7)
        plt.title(f'Histogram for Class: {classes[i]}')
        plt.xlabel('Token Index')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        outputpath = os.path.join(path, f"Histogram_Class_{classes[i]}.png")
        plt.savefig(outputpath)
        plt.close()
        print(f"Saved histogram for class {classes[i]} with shape {class_indices.shape}")



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
    
    # Model paths
    VQVAE_PATH = 'saved_models/vqvae_monai.pth'
    MONAI_COND_PATH = 'saved_models/MONAI_Cond2_Transformer_epochs_50.pt'
    MONAI_UNCOND_PATH = 'saved_models/MONAI_Transformer_epochs_50.pt'
    NANOGPT_COND_PATH = 'saved_models/NanoGPT_Cond2_Transformer_epochs_50.pt'
    NANOGPT_UNCOND_PATH = 'saved_models/NanoGPT_Transformer_epochs_50.pt'
    BOS_TOKEN = 64
    samples_per_class = 2016
    iq_samples = 1024
    device = 'cuda:0'

    
    # Create base evaluation folder
    base_eval_folder = 'EvaluationResults'
    os.makedirs(base_eval_folder, exist_ok=True)

    # Load all models
    VQVAE = torch.load(VQVAE_PATH).to(device)
    VQVAE.eval()
    
    monai_cond = torch.load(MONAI_COND_PATH).to(device)
    monai_uncond = torch.load(MONAI_UNCOND_PATH).to(device)
    nanogpt_cond = torch.load(NANOGPT_COND_PATH).to(device)
    nanogpt_uncond = torch.load(NANOGPT_UNCOND_PATH).to(device)
    
    all_models = {
        'MONAI_Conditional': (monai_cond, True),
        'MONAI_Unconditional': (monai_uncond, False),
        'NanoGPT_Conditional': (nanogpt_cond, True),
        'NanoGPT_Unconditional': (nanogpt_uncond, False)
    }

    # Load data
    train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
        classes=classes,
        iq_samples=iq_samples,
        samples_per_class=samples_per_class,
        batch_size=4
    )

    # Save real dataset if it doesn't exist
    real_dataset_path = "src/Saved_datasets/real_dataset.npy"
    if not os.path.exists(real_dataset_path):
        save_torch_dataset_as_numpy(ds_train, real_dataset_path)

    # Process each model
    for model_name, (model, is_conditional) in all_models.items():
        print(f"\nProcessing {model_name}...")
        
        # Create model-specific folders
        model_eval_folder = os.path.join(base_eval_folder, f'{model_name}_results')
        histogram_folder = os.path.join(model_eval_folder, 'CodebookHistograms')
        os.makedirs(model_eval_folder, exist_ok=True)
        os.makedirs(histogram_folder, exist_ok=True)
        
        # Generate reconstructions
        for j in range(10):
            all_reconstructions = []
            labels = list(range(len(classes)))
            
            for i in labels:
                if is_conditional:
                    # For conditional models, provide both BOS token and label
                    new_indices = model.generate(torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + i]]).to(device), max_new_tokens=512)
                    new_indices = new_indices[:,2:]
                else:
                    # For unconditional models, only provide BOS token
                    new_indices = model.generate(torch.tensor([[BOS_TOKEN]]).to(device), max_new_tokens=512)
                    new_indices = new_indices[:,1:]
                    
                new_indices = torch.tensor(new_indices, device=device)
                
                reconstruction = VQVAE.decode(VQVAE.codebook.lookup(new_indices))
                all_reconstructions.append(reconstruction.squeeze())

            all_reconstructions = torch.stack(all_reconstructions)
            plot_generated_reconstructions(all_reconstructions, labels, model_eval_folder, j, classes)

        # Generate fake dataset and compute metrics
        fake_dataset_path = f"src/Saved_datasets/fake_dataset_{model_name}.npy"
        fake_indices = generate_fake_dataset(
            transformer_model=model,
            is_conditional = is_conditional,
            vqvae_model=VQVAE,
            output_file=fake_dataset_path,
            num_labels=6, 
            BOS_TOKEN = 64, 
            samples_per_label=samples_per_class, 
            batch_size=32, 
            signal_length=1024
        )
        # Plot codebook histograms
        plot_codebook_histograms(fake_indices, histogram_folder)
        
        # Compute and print metrics
        metrics = compute_metrics(real_dataset_path, fake_dataset_path)
        print(f"Computed Metrics for {model_name}:", metrics)

    # Process test dataset
    test_histogram_path = os.path.join(base_eval_folder, "TestDatasetHistograms")
    indices = quantize_dataset(VQVAE, ds_test)
    plot_codebook_histograms(indices, test_histogram_path)