import torch
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from src.Dataset import getDataLoader
from torchsig.utils.cm_plotter import plot_confusion_matrix
from torch import nn

def eval_classifier(classes: list, classifier, ds_test: ModulationsDataset, transformer_model=None, vqvae_model=None, num_samples=500):
    accuracies = []
    classifier.to(device).eval()
    if transformer_model is not None and vqvae_model is not None:
        # Fake data generation incorporated inside the classifier evaluation
        transformer_model.to(device).eval()
        vqvae_model.to(device).eval()
        fake_signals = generate_fake_data(transformer_model, vqvae_model, classes, num_samples)
    elif(transformer_model is None and vqvae_model is not None):
        vqvae_model.to(device).eval()
        fake_signals = []
        for i in range(len(ds_test)):
            input_data = torch.from_numpy(ds_test[i][0]).to(device)
            # Add batch dimension and ensure shape is [1, 2, 1024]
            if len(input_data.shape) == 2:
                input_data = input_data.unsqueeze(0)
            reconstructed = vqvae_model.reconstruct(input_data)
            # Remove batch dimension for the classifier
            fake_signals.append((reconstructed.squeeze(0).cpu().numpy(), ds_test[i][1]))
    else:
        # Use the provided dataset if no model for generation is provided
        fake_signals = [(ds_test[i][0], ds_test[i][1]) for i in range(len(ds_test))]

    y_preds = np.zeros(len(fake_signals))
    y_true = np.zeros(len(fake_signals))

    for i, (data, label) in tqdm(enumerate(fake_signals)):
        test_x = torch.from_numpy(np.expand_dims(data, axis=0)).float().to(device)
        output = classifier(test_x)
        pred_label = output.argmax(dim=1, keepdim=True)
        y_preds[i] = pred_label.item()
        y_true[i] = label

    acc = (y_preds == y_true).sum().item() / len(fake_signals)
    accuracies.append(acc * 100)

    print(f"Accuracy: {acc * 100:.2f}%")
    plot_confusion_matrix(
        y_true,
        y_preds,
        classes=classes,
        normalize=True,
        title="Confusion Matrix\nTotal Accuracy: {:.2f}%".format(acc * 100),
        text=False,
        rotate_x_text=60,
        figsize=(10, 10)
    )
    
    confusionMatrix_save_path = "./EvaluationResults/Eff_Net_Confusion_Matrix.png"
    if transformer_model is not None and vqvae_model is not None:
        print(str(type(transformer_model)))
        if "Monai" in str(type(transformer_model)):
            confusionMatrix_save_path = "./EvaluationResults/MONAI_Cond2_Confusion_Matrix.png"
        elif "TransformerModel" in str(type(transformer_model)):
            confusionMatrix_save_path = "./EvaluationResults/NanoGPT_Cond2_Confusion_Matrix.png"
    elif transformer_model is None and vqvae_model is not None:
        confusionMatrix_save_path = "./EvaluationResults/VQVAE_Confusion_Matrix.png"
    
    
    plt.savefig(confusionMatrix_save_path)
    plt.close()
    print("\nClassification Report:")
    print(classification_report(y_true, y_preds, target_names=classes))

    return accuracies

def generate_fake_data(transformer_model, vqvae_model, classes, num_samples):
    BOS_TOKEN = 64  # As specified
    fake_signals = []
    
    for label in range(len(classes)):
        # Modified label tensor generation as per specification
        label_tensor = torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + label]]).to(device)
        
        new_indices = transformer_model.generate(
            label_tensor,
            max_new_tokens=1024 // 2
        )
        
        reconstructions = vqvae_model.decode(vqvae_model.codebook.lookup(new_indices[:, 2:])).detach()
        for rec in reconstructions:
            reshaped_rec = rec.view(2, 1024)
            fake_signals.append((reshaped_rec.cpu().numpy(), label))
    
    return fake_signals

if __name__ == "__main__":
    classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]
    device = 'cuda:0'
    
    # Load models from the saved_models directory
    model = torch.load('./saved_models/efficientNet_epochs_10.pt',weights_only=False).to(device)
    model.eval()
    
    vqvae_model = torch.load('./saved_models/vqvae_monai.pth',weights_only=False).to(device)
    vqvae_model.eval()
    
    # Load only the conditional (cond2) transformer models
    monai_transformer = torch.load('./saved_models/MONAI_Cond2_Transformer_epochs_50.pt',weights_only=False)
    monai_transformer.eval()
    
    nanogpt_transformer = torch.load('./saved_models/NanoGPT_Cond2_Transformer_epochs_50.pt',weights_only=False)
    nanogpt_transformer.eval()

    # Load data
    dl_train, ds_train, dl_test, ds_test, dl_val, ds_val = getDataLoader(
        classes=classes,
        iq_samples=1024,
        samples_per_class=1000,
        batch_size=4
    )

    print("\nEvaluating Efficient net:")
    accuracies_effNet = eval_classifier(
        classes, 
        model, 
        ds_test,
    )
    print(f"EfficientNet Accuracies: {accuracies_effNet}")


    print("\nEvaluating VQVAE reconstructions:")
    accuracies_vqvae = eval_classifier(
        classes, 
        model, 
        ds_test,
        vqvae_model=vqvae_model
    )
    print(f"VQVAE reconstructions Accuracies: {accuracies_vqvae}")


    # Evaluate with MONAI conditional transformer
    print("\nEvaluating MONAI Conditional Transformer:")
    accuracies_monai = eval_classifier(
        classes, 
        model, 
        ds_test,
        transformer_model=monai_transformer,
        vqvae_model=vqvae_model
    )
    print(f"MONAI Conditional Transformer Accuracies: {accuracies_monai}")

    # Evaluate with NanoGPT conditional transformer
    print("\nEvaluating NanoGPT Conditional Transformer:")
    accuracies_nanogpt = eval_classifier(
        classes, 
        model, 
        ds_test,
        transformer_model=nanogpt_transformer,
        vqvae_model=vqvae_model
    )
    print(f"NanoGPT Conditional Transformer Accuracies: {accuracies_nanogpt}")