import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONetCP(nn.Module):
    def __init__(self, branch_layer, trunk_layer,activation):
        super(DeepONetCP, self).__init__()
        self.branch = DenseNet(branch_layer, activation)
        self.trunk = DenseNet(trunk_layer, activation)

    def forward(self, u0, grid):
        u0 = u0.permute(0,2,3,1).reshape(-1,4)
        a = self.branch(u0)
        b = self.trunk(grid) 
        output = torch.einsum('bi,ni->bn', a, b)
        yy = torch.exp(output) 
        return yy
        # single deepOnet map distribution
            # branch input (B*X*Y,4); (samples,(ux,uy,T0,rho0))
            # trunk input (9,2); (directions,(ex,ey))
            # branch output (B*X*Y,50); projection of macroscopic features
            # trunk output (9,50); higher-dimentional basis

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
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return x
    
if __name__ == "__main__":
    # Define the model
    branch_layer = [4, 50, 50, 50, 50]
    trunk_layer = [2, 50, 50, 50, 50]
    activation = 'relu'
    model = DeepONetCP(branch_layer, trunk_layer, activation)
    print(model)