from src.VQVAE import VQVAE
from src.utils import *
from src.Dataset import getDataLoader
import lightning.pytorch as pl
import torch

def train_VQVAE(dl_train,
                    dl_val,
                    epochs = epochs,
                    train_bool= True,
                    eff_net_PATH = eff_net_PATH,
                    classes = classes,
                    in_channels = in_channels):
    
    Trainer = pl.Trainer(max_epochs=epochs,logger=None, devices=devices, accelerator = 'gpu')
    vqvae_model = VQVAE(in_feat_dim=in_channels,
                        hidden_dim=enc_hidden_dim,
                        out_feat_dim=dec_hidden_dim,
                        num_res_blocks=num_res_blocks,
                        codebook_dim=codebook_dim, 
                        codebook_slots= codebook_slots)
    Trainer.fit(vqvae_model,dl_train,dl_val)
    return vqvae_model


if __name__ == '__main__':

    dl_train, ds_train, dl_test, ds_test, dl_val, ds_val = getDataLoader(
        classes = classes,
        iq_samples = iq_samples,
        samples_per_class= samples_per_class,
        batch_size=batch_size
    )
    print(len(dl_train), len(dl_val))
    print("Started Training Classifier")
    model = train_VQVAE(
                    dl_train= dl_train,
                    dl_val=dl_val,
                    epochs = epochs,
                    train_bool= True,
                    eff_net_PATH = eff_net_PATH,
                    classes = classes,
                    in_channels = in_channels
                    )
    torch.save(model,VQVAE_PATH)
    print("Trained Classifier")