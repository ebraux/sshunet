
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DoubleConv(nn.Module):
    """Double convolution block for U-NET"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SSHDataset(Dataset):
    """Sea Surface Height dataset loader"""
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the SSH data files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.transform = transform
        # Load your data - this is a placeholder
        # In a real implementation, you would load your netCDF or HDF5 files here
        self.data = self._load_data()
        
    def _load_data(self):
        # Placeholder for data loading
        # In a real implementation, you would use libraries like xarray or h5py
        # Example: return xr.open_dataset(self.data_path)
        # For now, returning random data for demonstration
        return np.random.randn(100, 1, 256, 256)  # 100 samples of 256x256 SSH maps
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get SSH data
        ssh_map = self.data[idx]
        
        # In a real implementation, you might have input/target pairs
        # For now, we're using the same data as input and target (like denoising)
        sample = {'input': ssh_map, 'target': ssh_map}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SSHUNetModel(pl.LightningModule):
    def __init__(self, input_channels=1, output_channels=1, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(n_channels=input_channels, n_classes=output_channels)
        self.learning_rate = learning_rate
        
        # Define loss function - MSE for regression tasks like SSH prediction
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log('test_loss', test_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def train_ssh_model():
    # Create model
    model = SSHUNetModel()
    
    # Create datasets
    train_dataset = SSHDataset(data_path='path/to/train/data')
    val_dataset = SSHDataset(data_path='path/to/val/data')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model

if __name__ == "__main__":
    trained_model = train_ssh_model()
