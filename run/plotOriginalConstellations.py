import os
import matplotlib.pyplot as plt
import torch
from src.Dataset import getDataLoader
from src.utils import *
from tqdm import tqdm
def plot_signals_by_class(data_loader, classes, folder, vqvae_model=None):
    os.makedirs(folder, exist_ok=True)
    class_signals = {cls: None for cls in range(len(classes))}

    # Ensure the model is in evaluation mode and moved to the correct device
    if vqvae_model:
        vqvae_model = vqvae_model.to(device).eval()

    for signals, labels in tqdm(data_loader):
        signals = signals.to(device)  # Move signals to GPU if using CUDA
        for signal, label in zip(signals, labels):
            label_idx = label.item()
            if class_signals[label_idx] is None:
                if vqvae_model:
                    with torch.no_grad():
                        # Ensure signal has correct dimensions: (batch_size, channels, length)
                        signal = signal.unsqueeze(0)  # Add batch dimension if it's missing
                        reconstructed = vqvae_model.reconstruct(signal)
                        class_signals[label_idx] = reconstructed.squeeze(0).cpu().numpy()  # Remove batch dimension for plotting
                else:
                    class_signals[label_idx] = signal.cpu().numpy()

        if all(val is not None for val in class_signals.values()):
            break

    plt.figure(figsize=(18, 6 * ((len(classes) + 2) // 3)))
    for i, (label_idx, signal) in enumerate(class_signals.items()):
        ax = plt.subplot((len(classes) + 2) // 3, 3, i + 1)
        ax.scatter(signal[0], signal[1], alpha=0.7, label=classes[label_idx])
        ax.set_title(f'{classes[label_idx]}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'signals_by_class.png'))
    plt.close()

# Load your VQVAE model correctly and ensure it's ready for inference
vqvae_model = torch.load(VQVAE_PATH).to(device).eval()

# Define your data loader and classes
train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
    classes=classes,
    iq_samples=1024,
    samples_per_class=1000,
    batch_size=6
)

# Plot the signals
plot_signals_by_class(train_dl, classes, "EvaluationResults")
