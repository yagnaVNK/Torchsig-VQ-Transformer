
from torch.utils.data import DataLoader
import torch
from torchsig.models.iq_models.efficientnet.efficientnet import  create_effnet
import lightning.pytorch as pl
import timm
from src.Classifier import ExampleNetwork
from src.utils import *
from pytorch_lightning import  Trainer
from src.Dataset import getDataLoader
import torch.nn as nn
from timm.layers.norm_act import BatchNormAct2d
from torchsummary import summary

def replace_batchnormact2d_with_batchnorm1d(module):
    """
    Recursively replace all BatchNormAct2d layers with nn.BatchNorm1d layers in the given module.
    """
    for name, child in module.named_children():
        # If the layer is BatchNormAct2d, replace it
        if isinstance(child, BatchNormAct2d):  
            new_layer = nn.BatchNorm1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            setattr(module, name, new_layer)
        else:
            # Recurse into child modules
            replace_batchnormact2d_with_batchnorm1d(child)

def train_classifier(dl_train: DataLoader,
                     dl_val: DataLoader,
                     epochs: int,
                     train_bool: bool,
                     eff_net_PATH: str,
                     classes: list,
                     in_channels: int) -> ExampleNetwork:
    
    model = create_effnet(
        timm.create_model(
            "efficientnet_b4",
            num_classes=len(classes),
            in_chans=in_channels,  
        )
    )
    
    example_model = ExampleNetwork(model, dl_train, dl_val)
    example_model = example_model.float().to(device)

    replace_batchnormact2d_with_batchnorm1d(example_model)
    #print(example_model)

    trainer = Trainer(
        max_epochs=epochs,
        devices=devices,
        accelerator="gpu"
    )
    
    if train_bool:
        #print(summary(example_model.to(device),(2,1024),16,device=device))
        trainer.fit(example_model, dl_train, dl_val)
        torch.save(example_model.state_dict(), eff_net_PATH)
        print("trained the model")
        return example_model
    else:
        example_model.load_state_dict(torch.load(eff_net_PATH))
        print("loaded from checkpoint")
        return example_model
    
if __name__ == '__main__':

    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    iq_samples = 1024
    samples_per_class= 1000
    batch_size = 32
    epochs = 10
    eff_net_PATH = f'src/SavedModels/efficientNet_epochs_{epochs}.pt'
    in_channels = 2

    dl_train, ds_train, dl_test, ds_test, dl_val, ds_val = getDataLoader(
        classes = classes,
        iq_samples = iq_samples,
        samples_per_class= samples_per_class,
        batch_size=batch_size
    )
    print(len(dl_train), len(dl_val))
    print("Started Training Classifier")
    model = train_classifier(
                    dl_train= dl_train,
                    dl_val=dl_val,
                    epochs = epochs,
                    train_bool= True,
                    eff_net_PATH = eff_net_PATH,
                    classes = classes,
                    in_channels = in_channels
                    )
    print("Trained Classifier")