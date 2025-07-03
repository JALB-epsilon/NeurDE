import torch
import torch.nn as nn

class NeurDE(nn.Module):
    def __init__(self, alpha_layer, phi_layer, activation):
        super(NeurDE, self).__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(phi_layer, activation)

    def forward(self, u0, grid):
        # Ensure consistent input shapes
        u0 = u0.permute(0, 2, 3, 1).reshape(u0.size(0) * u0.size(2) * u0.size(3), 4)
        a = self.alpha(u0)
        b = self.phi(grid)
        output = torch.einsum('bi,ni->bn', a, b)
        yy = torch.exp(output)
        return yy

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity.lower() == 'tanh':
                self.nonlinearity = nn.Tanh()
            elif nonlinearity.lower() == 'relu':
                self.nonlinearity = nn.ReLU()
            else:
                raise ValueError(f'{nonlinearity} type {type(nonlinearity)} is not supported')

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    
if __name__ == "__main__":
    # Define the model
    alpha_layer = [4, 50, 50, 50, 50]
    phi_layer = [2, 50, 50, 50, 50]
    activation = 'relu'
    model = NeurDE(alpha_layer, phi_layer, activation)
    print(model)