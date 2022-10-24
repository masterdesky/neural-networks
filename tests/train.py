#!/usr/bin/env python3
from ..autoencoders.dataset import CustomDataset
from ..autoencoders.denseautoencoder import DenseAutoEncoder
from ..autoencoders.conv1dautoencoder import Conv1DAutoEncoder
from ..autoencoders.conv2dautoencoder import Conv2DAutoEncoder


def train(model, data_train):
    pass

def predict(model, data_test):
    pass

def main():
    
    data_path = '../data/mnist_{}_data.csv'
    labels_path = '../data/mnist_{}_labels.csv'
    data_train = CustomDataset(data_path.format('train'), labels_path.format('train'))
    data_test = CustomDataset(data_path.format('test'), labels_path.format('test'))
    
    model = Conv1DAutoEncoder(input_size=4096)

    train(model, data_train)
    predict(model, data_test)

if __name__ == "__main__":
    main()
