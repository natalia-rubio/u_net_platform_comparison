import torch
import torch.nn as nn
import copy
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import time


class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, pool_size: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, kernel_size: int, pool_size: int, padding: int):
        super().__init__()
        self.convT = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=pool_size, stride=2)
        self.conv = nn.Conv2d(hidden_channels+out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.convT(x)
        x = self.conv(torch.cat([x, skip_connection], dim=1))
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self,
    hidden_channels: int = 2,
    num_levels: int = 2,
    kernel_size: int = 3,
    pool_size: int = 2,
    padding: int = 1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_levels):
            if i == 0:
                in_channels = 1
            else:
                in_channels = self.hidden_channels
            self.encoder_layers.append(EncoderLayer(in_channels, self.hidden_channels, self.kernel_size, self.pool_size, self.padding))

        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_levels):
            if i == self.num_levels - 1:
                out_channels = 1
            else:
                out_channels = self.hidden_channels
            self.decoder_layers.append(DecoderLayer(self.hidden_channels, out_channels, self.kernel_size, self.pool_size, self.padding))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = [x,]
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x.clone()) # save for skip connections
        for i, layer in enumerate(self.decoder_layers): 
            x = layer(x, skip_connections[-i-2])
        return x

    def loss_function(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target)

def test_unet():
    unet = UNet()
    x = torch.randn(1, 1, 256, 256)
    print(unet(x).shape)

def load_data():

    images = torch.load("data/images.pt")
    labels = torch.load("data/labels.pt")
    perm = torch.randperm(len(images))
    train_indices = perm[:int(len(images) * 0.8)]
    test_indices = perm[int(len(images) * 0.8):]
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    return train_images, train_labels, test_images, test_labels

def train_model(use_gpu: bool = False,
                num_epochs: int = 5,    
                batch_size: int = 16,
                num_batches: int = 16,
                kernel_size: int = 3,
                pool_size: int = 2):

    print(f"Training model with kernel size {kernel_size}, batch size {batch_size}, num batches {num_batches}")
    train_images, train_labels, test_images, test_labels = load_data()
    train_images = train_images[num_batches * batch_size:]
    train_labels = train_labels[num_batches * batch_size:]
    test_images = test_images[:batch_size]
    test_labels = test_labels[:batch_size]
    n_total = len(train_images) + len(test_images)
    print(
        f"Dataset from data/images.pt: {n_total} samples total; "
        f"train {len(train_images)} (80%), test {len(test_images)} (20%). "
        f"Tensor shape {train_images.shape}"
    )

    unet = UNet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    print(train_images.shape, train_labels.shape)

    if use_gpu:
        unet.to("cuda")
        train_images = train_images.to("cuda")
        train_labels = train_labels.to("cuda")
        test_images = test_images.to("cuda")
        test_labels = test_labels.to("cuda")
        print("Using GPU")

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=len(test_images), shuffle=False)

    epoch_train_loss = unet.loss_function(unet(train_images), train_labels)
    epoch_test_loss = unet.loss_function(unet(test_images), test_labels)
    print(f"Initial:  train loss: {epoch_train_loss.item()}  |  test loss: {epoch_test_loss.item()}")

    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = unet(images)
            loss = unet.loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_train_loss = unet.loss_function(unet(train_images), train_labels)
        epoch_test_loss = unet.loss_function(unet(test_images), test_labels)
        print(f"Epoch {epoch}:  train loss: {epoch_train_loss.item()}  |  test loss: {epoch_test_loss.item()}")
    end_time = time.perf_counter()
    training_time = end_time - start_time
    print(f"Time taken: {training_time} seconds")
    return training_time


if __name__ == "__main__":
    #test_unet()
    train_model()