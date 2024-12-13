
from torch.utils.data import DataLoader
import torch
from torchsig.models.iq_models.efficientnet.efficientnet import  create_effnet
import lightning.pytorch as pl
from src.utils import *
from pytorch_lightning import  Trainer
import torch.nn as nn
from torchsummary import summary
import os
from src.Classifier import SimpleCNN1D
from torchsig.datasets.modulations import ModulationsDataset
from src.utils import *
import torchsig.transforms as ST



batch_size = 64
data_transform = ST.Compose([
        ST.ComplexTo2D(),
    ])
ds_train = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=iq_samples,
        num_samples=int(len(classes) * samples_per_class),
        include_snr=False,
        transform=data_transform
    )

dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
ds_val = ModulationsDataset(
    classes=classes,
    use_class_idx=True,
    level=0,
    num_iq_samples=iq_samples,
    num_samples=int(len(classes) * samples_per_class // 10),  
    include_snr=False,
    transform=data_transform
)

dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
model = SimpleCNN1D(dl_train,dl_val,num_classes=6)
trainer = Trainer(
    max_epochs=10,
    devices=[0,1]
)

trainer.fit(model)
torch.save(model,eff_net_PATH)
