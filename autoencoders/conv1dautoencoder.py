import torch
import torch.nn as nn

from .baseautoencoder import BaseModel


class Conv1DAutoEncoder(BaseModel):
    def __init__(self, input_size : int, n_channels : int,
                 *,
                 kernel_size : int=21,
                 stride : int=1,
                 padding : int=0,
                 dilation : int=1,
                 hidden_dims : list=[32, 64, 128]):
        super(Conv1DAutoEncoder, self).__init__()

        assert kernel_size <= input_size, "Kernel size should not be larger than the input size!"
        assert kernel_size >= 1, "Kernel size should be at least 1 pixel!"
        assert stride >= 1, "Stride length should be at least 1 pixel!"
        assert padding >= 0, "Width of padding should be non-negative!"
        assert dilation >= 1, "Dilation should be at least 1 pixel!"
        assert len(hidden_dims) >= 1, "There should be at least 1 hidden layer!"

        self.input_size = input_size
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.hidden_dims = hidden_dims

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):
        encoder = []
        n_channels = self.n_channels
        hidden_dims = self.hidden_dims[:]
        for h_dim in hidden_dims:
            encoder.append(torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=n_channels, out_channels=h_dim,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation),
                torch.nn.ReLU())
            )
            n_channels = h_dim
        encoder = torch.nn.Sequential(*encoder)
        return encoder

    def create_decoder(self):
        decoder = []
        hidden_dims = self.hidden_dims[::-1]
        for i, _ in enumerate(hidden_dims[:-1]):
            decoder.append(torch.nn.Sequential(
                torch.nn.ConvTranspose1d(
                    in_channels=hidden_dims[i], out_channels=hidden_dims[i+1],
                    kernel_size=self.kernel_size, stride=2*self.stride,
                    padding=self.padding, dilation=self.dilation),
                torch.nn.ReLU())
            )
        final_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=hidden_dims[-1], out_channels=hidden_dims[-1],
                kernel_size=self.kernel_size, stride=self.stride,
                padding=self.padding, dilation=self.dilation),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=hidden_dims[-1], out_channels=self.n_channels,
                kernel_size=self.kernel_size, stride=self.stride,
                padding=self.padding, dilation=self.dilation),
            torch.nn.Sigmoid()
        )
        decoder = torch.nn.Sequential(*decoder, final_layer)
        return decoder

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))

    def generate(self, latent_layer):
        return self.decoder(latent_layer)