#!/usr/bin/env python3
import sys

import torch
import torchvision

from ..dataset.dataset import CustomDataset
from ..autoencoders.denseautoencoder import DenseAutoEncoder
from ..autoencoders.conv1dautoencoder import Conv1DAutoEncoder
from ..autoencoders.conv2dautoencoder import Conv2DAutoEncoder


def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Initialize the gradients as zero
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    # Save model
    PATH = '../models/imagenet.pth'
    torch.save(model.state_dict(), PATH)

def predict(model, test_loader):
    pass

def main():
    train_data = torchvision.datasets.ImageNet(
        '../data/imagenet', train=True, download=True)
    test_data = torchvision.datasets.ImageNet(
        '../data/imagenet', train=False, download=True)

    # Class that inherits `torch.utils.data.Dataset` and contains a __len__ and
    # __getitem__ method, specific for the dataset 
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=4, shuffle=True, num_workers=4)

    input_size = torch.tensor(train_data[0][0].size()).prod()
    model = DenseAutoEncoder(input_size=input_size)

    #train(model, train_loader)
    #predict(model, test_loader)

if __name__ == "__main__":
    main()
