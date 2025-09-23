import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    A neural network layer that uses the sine function as its activation.
    It's designed for use in a SIREN model and includes the specific weight
    initialization critical for SIRENs to work.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 is_first: bool = False,
                 omega_0: float = 30.0):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.is_first = is_first
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Initialization for the first layer
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                # Initialization for subsequent layers
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        # Apply the sine activation function
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    A full SIREN model, which is a stack of SineLayers.
    """
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 first_omega_0: float = 30.0,
                 hidden_omega_0: float = 30.0):

        super().__init__()

        self.net = []
        # First layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        # Final layer - typically a standard linear layer
        final_linear = nn.Linear(hidden_features, out_features, bias=True)

        # Initialize final layer weights like a standard SIREN layer
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
            if final_linear.bias is not None:
                final_linear.bias.zero_()

        self.net.append(final_linear)

        # Make the list of layers a sequential module
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # The input coordinates should be scaled to the range [-1, 1]
        return self.net(coords)


class SineLayerWithShift(nn.Module):
    def __init__(self,
                 in_features: int,
                 latent_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 is_first: bool = False,
                 omega_0: float = 30.0,
                 shift_phi=None):

        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.is_first = is_first
        self.omega_0 = omega_0
        self.shift_phi = shift_phi

        self.x_linear = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)
        self.phi_linear = nn.Linear(latent_features, out_features, bias=self.use_bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Initialization for the first layer
                self.x_linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                # Initialization for subsequent layers
                self.x_linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

            # Initialization for shift layers, using Haiku default
            self.phi_linear.weight.uniform_(-np.sqrt(1. / self.in_features), np.sqrt(1. / self.in_features))


    def forward(self, x):
        # Apply the sine activation function with shift
        return torch.sin(self.omega_0 * (self.x_linear(x) + self.phi_linear(self.shift_phi)))


class SIRENWithShift(nn.Module):
    def __init__(self,
                 in_features: int,
                 latent_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 first_omega_0: float = 30.0,
                 hidden_omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = []

        # First layer
        self.net.append(SineLayerWithShift(in_features, latent_features, hidden_features,
                                           is_first=True, omega_0=first_omega_0))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayerWithShift(hidden_features, latent_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

        # Final layer - typically a standard linear layer
        self.final_linear = nn.Linear(hidden_features, out_features, bias=True)

        # Initialize final layer weights like a standard SIREN layer
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                         np.sqrt(6 / hidden_features) / hidden_omega_0)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

        # self.net.append(final_linear)

    def forward(self, x, phi):
        # The input coordinates should be scaled to the range [-1, 1]
        for layer in self.net:
            layer.shift_phi = phi
        x = self.net(x)
        out = self.final_linear(x)
        return out
