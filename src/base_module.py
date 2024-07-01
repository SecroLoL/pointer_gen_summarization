"""
Base module for all layers in the pointer-gen network
"""

import torch
import torch.nn as nn


class BaseModule(nn.Module):

    def __init__(self, initialization: str):
        """
        Constructor for initializing a BaseModule instance

        Args:
            initialization (str): Type of weight initialization to use (e.g. 'uniform', 'xavier')
        """
        super(BaseModule, self).__init__()  
        self.initialization = initialization

    def init_params(self):
        """
        Initialize weights for module. Currently supports uniform initialization
        """
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                if self.init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                    