import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
from pytorch_lightning import  Trainer

class SimpleCNN1D(LightningModule):
    def __init__(self, train_dl, val_dl, num_classes=6, learning_rate=0.001):
        super(SimpleCNN1D, self).__init__()
        self.save_hyperparameters()

        # Model layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.val_dl = val_dl
        self.train_dl = train_dl

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x.float())
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        #print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x.float())
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        #print(f"val loss : {loss}")
        return {'val_loss': loss}

    def train_dataloader(self):
        # Define your training DataLoader here
        return self.train_dl

    def val_dataloader(self):
        # Define your validation DataLoader here
        return self.val_dl


