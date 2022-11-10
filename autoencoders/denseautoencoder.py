import torch

from .baseautoencoder import BaseModel

class DenseAutoEncoder(BaseModel):

    def __init__(self, input_size : int,
                 *,
                 levels : int=4,
                 latent_dim : int=16,
                 hidden_neurons=None):
        super(DenseAutoEncoder, self).__init__()

        if hidden_neurons is not None:
            assert len(hidden_neurons) >= 1, "There should be at least 1 hidden layer!"
        else:
            assert levels >= 1, "There should be at least 1 hidden layer!"
            assert latent_dim >=1, "There should be at least 1 neuron in the latent space!"

        self.input_size = input_size
        self.levels = levels
        self.latent_dim = latent_dim
        self.hidden_neurons = hidden_neurons

        if self.hidden_neurons is None:
            # Determine the numbers of neurons in each layer
            self.hidden_neurons = torch.linspace(
                self.input_size, self.latent_dim, self.levels, 
                dtype=torch.int16
            )
        else:
            self.hidden_neurons = torch.tensor(self.hidden_neurons, dtype=torch.int16)

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):
        encoder = []
        h_dim = self.hidden_neurons
        for prev, next in zip(h_dim, h_dim[1:]):
            encoder.append(torch.nn.Sequential(
                torch.nn.Linear(prev, next),
                torch.nn.ReLU())
            )
        encoder = torch.nn.Sequential(*encoder)
        return encoder

    def create_decoder(self):
        decoder = []
        h_dim = torch.flip(self.hidden_neurons, [0])
        for prev, next in zip(h_dim, h_dim[1:]):
            decoder.append(torch.nn.Sequential(
                torch.nn.Linear(prev, next),
                torch.nn.ReLU())
            )
        final_layer = torch.nn.Sequential(
            torch.nn.Linear(h_dim[-1], h_dim[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim[-1], self.input_size),
            torch.nn.Sigmoid()
        )
        decoder = torch.nn.Sequential(*decoder, final_layer)
        return decoder

    def forward(self, input_layer):
        return self.decoder(self.encoder(input_layer))

    def generate(self, latent_layer):
        return self.decoder(latent_layer)