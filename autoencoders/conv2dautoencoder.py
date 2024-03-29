import torch
import torch.nn as nn

from .baseautoencoder import BaseModel

class Conv2DAutoEncoder(BaseModel):

    def __init__(self, input_size, n_channels,
                 *,
                 kernel_size : int=3,
                 stride : int=1,
                 padding : int=0,
                 dilation : int=1):
        super(Conv2DAutoEncoder, self).__init__()

        assert kernel_size <= input_size, "Kernel size should not be larger than the input size!"
        assert kernel_size >= 1, "Kernel size should be at least 1 pixel!"
        assert stride >= 1, "Stride length should be at least 1 pixel!"
        assert padding >= 0, "Width of padding should be non-negative!"
        assert dilation >= 1, "Dilation should be at least 1 pixel!"

        self.input_size = input_size
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def encoder(self, input_layer):
        encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 128,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.Conv2d(128, 64,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.Conv2d(64, 32,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU()
        )
        return encoder(input_layer)

    def decoder(self, latent_layer):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.n_channels,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
            nn.Sigmoid()
        )
        return decoder(latent_layer)

    def forward(self, input_layer):
        return self.decoder(self.encoder(input_layer))

    def generate(self, latent_layer):
        return self.decoder(latent_layer)