import torch
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from src.utils import *
from src.Dataset import getDataLoader
from torchsig.utils.cm_plotter import plot_confusion_matrix
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def eval_classifier(classes: list, classifier, ds_test: ModulationsDataset, transformer_model=None, vqvae_model=None, num_samples=500):
    accuracies = []
    classifier.to(device).eval()
    if transformer_model is not None and vqvae_model is not None:
        # Fake data generation incorporated inside the classifier evaluation
        transformer_model.to(device).eval()
        vqvae_model.to(device).eval()
        fake_signals = generate_fake_data(transformer_model, vqvae_model, classes, num_samples)
    else:
        # Use the provided dataset if no model for generation is provided
        fake_signals = [(ds_test[i][0], ds_test[i][1]) for i in range(len(ds_test))]

    y_preds = np.zeros(len(fake_signals))
    y_true = np.zeros(len(fake_signals))

    for i, (data, label) in tqdm(enumerate(fake_signals)):
        test_x = torch.from_numpy(np.expand_dims(data, axis=0)).float().to(device)  # Ensure the data is [1, 2, 1024]
        output = classifier(test_x)
        pred_label = output.argmax(dim=1, keepdim=True)
        y_preds[i] = pred_label
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
        confusionMatrix_save_path = "./EvaluationResults/Transformer_model_Confusion_Matrix.png"
    elif transformer_model is None and vqvae_model is not None:
        confusionMatrix_save_path = "./EvaluationResults/vqvae_Confusion_Matrix.png"

    plt.savefig(confusionMatrix_save_path)
    plt.close()
    print("\nClassification Report:")
    print(classification_report(y_true, y_preds, target_names=classes))

    return accuracies
def generate_fake_data(transformer_model, vqvae_model, classes, num_samples):
    fake_signals = []
    for label in range(len(classes)):
        label_tensor = torch.tensor([[label]] * num_samples).to(device)
        new_indices = transformer_model.generate(
            label_tensor,
            max_new_tokens=1024 // 2  # Assuming each half represents one channel of complex data
        )
        reconstructions = vqvae_model.decode(vqvae_model.codebook.lookup(new_indices[:, 1:])).detach()
        for rec in reconstructions:
            # Ensure that rec is reshaped from [2048] to [2, 1024] if necessary
            reshaped_rec = rec.view(2, 1024)  # Adjust as necessary based on actual output
            fake_signals.append((reshaped_rec.cpu().numpy(), label))
    return fake_signals




if __name__ == "__main__":
    classes = ["4ask", "8pam", "16psk", "32qam_cross", "2fsk", "ofdm-256"]

    # Load the SimpleCNN1D model
    model = torch.load(eff_net_PATH).to(device)
    model.eval()
    vqvae_model = torch.load(VQVAE_PATH).to(device)
    vqvae_model.eval()
    transformer_model = torch.load(TRANSFORMER_MODEL_PATH)
    transformer_model.eval()
    monai_model = torch.load(MONAI_TRANSFORMER_MODEL_PATH)
    monai_model.eval()

    # Load data
    dl_train, ds_train, dl_test, ds_test, dl_val, ds_val = getDataLoader(
        classes=classes,
        iq_samples=1024,  # assuming each sample has 1024 IQ data points
        samples_per_class=1000,
        batch_size=4
    )

    #accuracies = eval_classifier(classes, model, ds_test)
    #print(accuracies)
    

    #accuracies = eval_classifier(classes, model, ds_test,vqvae_model)
    #print(accuracies)

    accuracies = eval_classifier(classes, model, ds_test,transformer_model=monai_model,vqvae_model=vqvae_model)
    print(accuracies)


