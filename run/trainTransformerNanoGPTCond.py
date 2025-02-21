

import torch
import torch.nn as nn 
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from src.Dataset import getDataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
from src.Transformer import *

if __name__ == '__main__':
    # Hyperparameters and setup
    epochs = 50
    TRANSFORMER_MODEL_PATH = f'saved_models/NanoGPT_Cond2_Transformer_epochs_{epochs}.pt'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    VQVAE_PATH = 'saved_models/vqvae_monai.pth'
    batch_size = 32
    num_workers = 4
    data_dir = '../Data'
    block_size = 512
    vocab_size = 64
    BOS_TOKEN = 64
    learning_rate = 1e-4
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    iq_samples = 1024
    samples_per_class= 1000

    # Load VQVAE model
    vqvae_model = torch.load(VQVAE_PATH).to(device)
    vqvae_model.eval()  # Ensure VQVAE is in eval mode

    # Initialize transformer model
    m = TransformerModel(
        vocab_size = 64,
        n_embd = 64, 
        block_size = 512,
        n_head = 8,
        n_layer = 8,
        additional_vocab=7
    ).to(device)


    # Data loading
    dl_train, ds_train, dl_test, ds_test, dl_val, ds_val = getDataLoader(
        classes = classes,
        iq_samples = iq_samples,
        samples_per_class= samples_per_class,
        batch_size=batch_size
    )

    # Process training data
    print("Processing training data...")
    train_sequences = []
    train_targets = []

    for x, label in tqdm(dl_train):
        x = x.to(device)
        label = label.to(device, dtype=torch.long)  # Ensure labels are long
        
        with torch.no_grad():
            vqvae_indices = vqvae_model.codebook.quantize_indices(vqvae_model.encode(x)).flatten(1)
        
        for i in range(len(label)):
            # Create input sequence: [BOS, LABEL+257, VQVAE_INDICES]
            input_seq = torch.cat([
                torch.tensor([BOS_TOKEN], device=device, dtype=torch.long),
                torch.tensor([BOS_TOKEN + 1  + label[i].item()], device=device, dtype=torch.long),
                vqvae_indices[i].long()  # Convert VQVAE indices to long
            ])
            
            # Create target sequence: [LABEL+257, VQVAE_INDICES, PAD]
            target_seq = torch.cat([
                torch.tensor([BOS_TOKEN + 1  + label[i].item()], device=device, dtype=torch.long),
                vqvae_indices[i].long(),
                torch.zeros(1, device=device, dtype=torch.long)
            ])
            
            train_sequences.append(input_seq)
            train_targets.append(target_seq)

    train_sequences = torch.stack(train_sequences)
    train_targets = torch.stack(train_targets)
    
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Process test data
    print("Processing test data...")
    test_sequences = []
    test_targets = []

    for x, label in tqdm(dl_test):
        x = x.to(device)
        label = label.to(device, dtype=torch.long)
        
        with torch.no_grad():
            vqvae_indices = vqvae_model.codebook.quantize_indices(vqvae_model.encode(x)).flatten(1)
        
        for i in range(len(label)):
            input_seq = torch.cat([
                torch.tensor([BOS_TOKEN], device=device, dtype=torch.long),
                torch.tensor([BOS_TOKEN + 1 + label[i].item()], device=device, dtype=torch.long),
                vqvae_indices[i].long()
            ])
            
            target_seq = torch.cat([
                torch.tensor([BOS_TOKEN + 1+ label[i].item()], device=device, dtype=torch.long),
                vqvae_indices[i].long(),
                torch.zeros(1, device=device, dtype=torch.long)
            ])
            
            test_sequences.append(input_seq)
            test_targets.append(target_seq)

    test_sequences = torch.stack(test_sequences)
    test_targets = torch.stack(test_targets)
    
    test_dataset = torch.utils.data.TensorDataset(test_sequences, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(m.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        m.train()
        total_train_loss = 0
        
        for input_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            # Move tensors to device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with both input and targets
            _, loss = m(input_batch, targets=target_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        m.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for input_batch, target_batch in test_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                
                _, loss = m(input_batch, targets=target_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    # Save model
    torch.save(m, TRANSFORMER_MODEL_PATH)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'Nano GPT loss curves {epochs} epochs.png')
    plt.close()

    # Generation example
    print("Generating example sequence...")
    label = 5
    context = torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + label]], dtype=torch.long, device=device)
    generated = m.generate(context, max_new_tokens=512)
    indices = generated[0, 2:]
    reshaped_indices = indices.view(1, 512)
    print("Generated indices shape:", reshaped_indices.shape)