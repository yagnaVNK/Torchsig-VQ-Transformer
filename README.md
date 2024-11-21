# Torchsig-VQ-Transformer

This repository contains the implementation of the Torchsig-VQ-Transformer model, including training scripts, evaluation utilities, saved models, and results. The project structure is organized to facilitate reproducibility and modularity.

---

## Project Structure

### Directories

- **run/EvaluationResults/Monai_results/**
  - Contains PNG files representing reconstruction results from the Monai Transformer model. File names indicate labels and reconstruction settings.
  - Example: `reconstruction_1_labels_0_to_5.png`.

- **src/SavedModels/**
  - This directory contains the pre-trained models required for training and evaluation:
    - `efficientNet_epochs_10.pt`
    - `MonaiTransformer_epochs_10.pt`
    - `Transformer_epochs_10.pt`
    - `VQVAE_epochs_10.pt`
  - **Note**: If the `SavedModels` folder does not exist, you must create it manually and download the pre-trained models as described below.

---

### Files

#### Python Scripts

- **Classifier.py**
  - Efficient Net classifier from torchsig 0.5.3 fine-tuned to classify the six classes of signal modulations. ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"] are the six classes used for the dataset and also the classifier.

- **Dataset.py**
  - Returns all the data loaders and datasets required for training and evaluation.

- **Transformer.py**
  - A GPT-2 model modified to generate 512 integers with a vocabulary between 0 and 63.

- **TransformerMonai.py**
  - A decoder-only transformer model from the `monai-generative` Python module. It also performs the same task as `Transformer.py`.

- **utils.py**
  - Contains configurations for training, model initialization, and evaluation.

- **VQVAE.py**
  - The vector quantized variational autoencoder (VQVAE) used to compress signal modulations and generate indices from the codebook for improved reconstructions.

- **Evaluate_models.py**
  - Evaluates the models and plots the reconstructions for both the Monai Transformer and GPT-2 Transformer.

- **train_Classifier.py**
  - Trains the classifier, which is used to compare the classification accuracy of the reconstructions with the original signals.

- **train_MonaiTransformer.py**
  - Trains the Monai-generative decoder-only transformer.

- **train_transformer.py**
  - Trains the GPT-2 model.

- **train_VQVAE.py**
  - Trains the VQVAE.

#### Visualizations

- **Monai_TransformerLossCurves.png**
  - Visualization of training loss curves for the Monai Transformer model.

- **TransformerLossCurves.png**
  - Visualization of training loss curves for the Transformer model.

#### Configuration and Metadata

- **.gitattributes**
  - Configuration file for defining attributes for the Git repository.

- **.gitignore**
  - Specifies files and directories to be ignored by Git.

- **README.md**
  - Documentation for the repository (this file).

- **requirements.txt**
  - List of Python dependencies required to run the project.

---

## Pre-trained Models

### Download Instructions

1. If the `SavedModels` folder does not exist, create it in the `src/` directory:
   ```bash
   mkdir -p src/SavedModels

2. Download the pre-trained models from the following Google Drive links:

    VQVAE model: [Download Here](https://drive.google.com/drive/folders/12Yud7KDqDubcSMn2MPEkxR2LdqzZ_sUm?usp=sharing)!
    Monai Transformer model: [Download Here](https://drive.google.com/drive/folders/12Yud7KDqDubcSMn2MPEkxR2LdqzZ_sUm?usp=sharing)!
    GPT-2 Transformer model: [Download Here](https://drive.google.com/drive/folders/12Yud7KDqDubcSMn2MPEkxR2LdqzZ_sUm?usp=sharing)!
    Efficient Net model: [Download Here](https://drive.google.com/drive/folders/12Yud7KDqDubcSMn2MPEkxR2LdqzZ_sUm?usp=sharing)!

3. After downloading, move all .pt files into the src/SavedModels directory.