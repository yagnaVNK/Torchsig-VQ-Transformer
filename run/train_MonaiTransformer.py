import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import *
from generative.networks.nets import DecoderOnlyTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.TransformerMonai import *
import matplotlib.pyplot as plt
from src.Dataset import getDataLoader



# Training loop
def train_monai(model, x_train_loader, y_train_loader,x_test_loader,y_test_loader, vocab_size=128, num_epochs=5, learning_rate=1e-4, device = device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loss=[]
    validation_loss = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(zip(x_train_loader, y_train_loader)):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb, yb[:, :-1])  # Feed input to predict next steps
            
            # Reshape output and target to match the input requirements of CrossEntropyLoss
            B, T, vocab_size = output.shape
            output = output.view(B * T, vocab_size)  # Flatten output to shape (B*T, vocab_size)
            target = yb[:, 1:].contiguous().view(-1)  # Flatten target to shape (B*T)

            # Ensure output and target match in size
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size]
                target = target[:min_size]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(x_train_loader))
        print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(x_train_loader)}')

        model.eval()
        total_loss = 0
        for xb, yb in zip(x_test_loader, y_test_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            output = model(xb, yb[:, :-1])  # Feed input to predict next steps
            
            # Reshape output and target to match the input requirements of CrossEntropyLoss
            B, T, vocab_size = output.shape
            output = output.view(B * T, vocab_size)  # Flatten output to shape (B*T, vocab_size)
            target = yb[:, 1:].contiguous().view(-1)  # Flatten target to shape (B*T)

            # Ensure output and target match in size
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size]
                target = target[:min_size]

            loss = criterion(output, target)
            total_loss += loss.item()
        validation_loss.append(total_loss / len(x_test_loader))
        print(f'Epoch {epoch + 1}, Validation Loss: {total_loss / len(x_test_loader)}')



    plt.figure(figsize=(10, 5)) 
    plt.plot(range(1,num_epochs+1), train_loss, 'bo-', label='Training Loss') 
    plt.plot(range(1,num_epochs+1), validation_loss, 'ro-', label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss over Epochs') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig("Monai_TransformerLossCurves.png")
    return model


if __name__ == "__main__":

    #device = 'cuda:1'
    # Load and preprocess the dataset
    train_dl, ds_train, test_dl, ds_test, val_dl, ds_val = getDataLoader(
        classes = classes,
        iq_samples = iq_samples,
        samples_per_class= samples_per_class,
        batch_size=batch_size
    )

    # Load VQVAE model
    vqvae_model = torch.load(VQVAE_PATH).to(device)

    # Quantize the dataset using VQVAE model
    data = torch.cat([torch.cat((label.unsqueeze(1).to(device), vqvae_model.codebook.quantize_indices(vqvae_model.encode(x.to(device)))), dim=1) for x, label in train_dl],dim=0)
    datasetlen = len(data)
    print(data[0].shape)
    print(data.shape)

    # Prepare input and target sequences for training
    x_train = torch.stack([data[i, :block_size] for i in range(len(data))])
    y_train = torch.stack([data[i, 1:block_size + 1] for i in range(len(data))])

    print(x_train.shape, y_train.shape)

    # Initialize model parameters
    d_model = 64
    nhead = 8
    num_layers = 8
    vocab_size = 64  # Numbers between 0 and 63
    max_len = 513
    batch_size = 32

    # Initialize model
    m = MonaiDecoderOnlyModel(d_model=d_model, nhead=nhead, num_layers=num_layers, vocab_size=vocab_size, max_len=max_len).to(device)
    m = torch.load(MONAI_TRANSFORMER_MODEL_PATH).to(device)


    optimizer = torch.optim.Adam(m.parameters(), lr = 5e-4)
    x_train_loader = DataLoader(x_train,batch_size=batch_size)
    y_train_loader = DataLoader(y_train,batch_size=batch_size)

    epochs = 500
    learning_rate = 1e-5
    m = m.to(device)


    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    test_data = torch.cat([torch.cat((label.unsqueeze(1).to(device), vqvae_model.codebook.quantize_indices(vqvae_model.encode(x.to(device)))), dim=1) for x, label in test_dl],dim=0)
    x_test = torch.stack([test_data[i,:block_size] for i in range(len(test_data))])
    y_test = torch.stack([test_data[i,1:block_size+1] for i in range(len(test_data))])
    x_test_loader = DataLoader(x_test,batch_size=batch_size)
    y_test_loader = DataLoader(y_test,batch_size=batch_size)
    
    m = train_monai(m,x_train_loader,y_train_loader,x_test_loader,y_test_loader, num_epochs=epochs,learning_rate = learning_rate)
    torch.save(m,MONAI_TRANSFORMER_MODEL_PATH)
    context = torch.tensor([[35],[42],[62]]).to(device=device)
    print(m.generate(context, max_new_tokens=513))

    


    

