import os
import matplotlib.pyplot as plt
import torch
from src.Dataset import getDataLoader

def plot_signals_by_class(data_loader, classes, folder):
    # Create the eval folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Initialize a dictionary to store one signal per class
    class_signals = {cls: None for cls in range(len(classes))}

    # Iterate through the data loader to fetch signals and their labels
    for signals, labels in data_loader:
        for signal, label in zip(signals, labels):
            label_idx = label.item()
            if class_signals[label_idx] is None:
                class_signals[label_idx] = signal

        # Break the loop if we have one signal for each class
        if all(val is not None for val in class_signals.values()):
            break

    # Create a figure with subplots
    cols = 3  # Number of columns in subplot grid
    rows = (len(classes) + cols - 1) // cols  # Calculate rows needed
    plt.figure(figsize=(18, 6 * rows))

    # Plot one signal for each class
    for i, (label_idx, signal) in enumerate(class_signals.items()):
        if signal is not None:
            ax = plt.subplot(rows, cols, i + 1)

            # Assuming signals are IQ samples with shape [2, n_samples]
            iq_signal = signal.cpu().detach().numpy()

            ax.scatter(iq_signal[0], iq_signal[1], alpha=0.7, label=classes[label_idx])
            ax.set_title(f'{classes[label_idx]}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.legend()

    # Save the plot
    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.savefig(os.path.join(folder, 'signals_by_class.png'))
    plt.close()

# Example usage
classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]
train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
    classes=classes,
    iq_samples=1024,  # Assuming IQ samples length is 1024
    samples_per_class=1000,  # Number of samples per class
    batch_size=6  # Batch size to fit 6 classes per plot
)

plot_signals_by_class(train_dl, classes, folder="EvaluationResults")
