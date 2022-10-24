import torch

class BaseModel(torch.nn.Module):
    '''
    Template model for the PyTorch implementation of encoder-decoder style
    architectures. Contains the necessary `forward()` and other optional 
    methods.

    Parameters:
    -----------
    None
    '''
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def encoder(self):
        raise NotImplementedError()

    def decoder(self):
        raise NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()

    def generate(self, latent_space):
        '''Naive autoencoders are not able to generate content!'''
        raise NotImplementedError()