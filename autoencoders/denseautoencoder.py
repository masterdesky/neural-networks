import torch
import torch.nn as nn

from .baseautoencoder import BaseModel

class DenseAutoEncoder(BaseModel):

    def __init__(self, input_size : int):
        super(DenseAutoEncoder, self).__init__()

        self.input_size = input_size

        # Determine the number of neurons in each layer
        self.n_size = (self.input_size-1).bit_length()
        self.n_filter = 1<<self.n_size
        self.layer_size = [self.input_size] + \
                          [self.n_filter>>(i+1) for i in range(self.n_size-6)]

    def encoder(self, input_layer):
        encoder = []
        for i in range(self.n_size-6):
            encoder.append(nn.Linear(self.input_size, self.n_filter>>(i+1)))
            encoder.append(nn.ReLU())
        encoder = nn.Sequential(*tuple(encoder))
        return encoder(input_layer)

    def decoder(self, latent_layer):
        decoder = []
        for i in reversed(range(self.n_size-6)):
            decoder.append(nn.Linear(self.layer_size[i+1], self.layer_size[i]))
            decoder.append(nn.ReLU())
        decoder[-1] = nn.Sigmoid()  # Last activation is Sigmoid
        decoder = nn.Sequential(*tuple(decoder))
        return decoder(latent_layer)

    def forward(self, input_layer):
        return self.decoder(self.encoder(input_layer))

    def generate(self, latent_layer):
        return self.decoder(latent_layer)