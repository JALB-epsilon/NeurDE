import torch
import torch.nn as nn

class GeneratingData(nn.Module):
    def __init__(self, device, Uax, Uay, Qn, X, Y):
        super(GeneratingData, self).__init__()
        self.device = device
        ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
        ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
        self.Uax = Uax
        self.Uay = Uay
        self.ex = torch.tensor(ex_values, dtype=torch.float32, device=self.device) + self.Uax
        self.ey = torch.tensor(ey_values, dtype=torch.float32, device=self.device) + self.Uay
        self.ex1 = torch.tensor(ex_values, dtype=torch.float32, device=self.device)
        self.ey1 = torch.tensor(ey_values, dtype=torch.float32, device=self.device)
        self.Qn = Qn
        self.X = X
        self.Y = Y
        self.get_basis()

    def get_basis(self):
        self.ex_ex = self.ex**2
        self.ey_ey = self.ey**2
        self.ex_plus_ey = self.ex + self.ey

    def forward(self, samples):
        # Generate samples with values between -0.1 and 0.1
        alpha00 = (torch.rand(samples, self.Y, self.X, device=self.device) * 0.2) - 0.1
        alpha10 = (torch.rand(samples, self.Y, self.X, device=self.device) * 0.2) - 0.1
        alpha01 = (torch.rand(samples, self.Y, self.X, device=self.device) * 0.2) - 0.1
        alpha11 = (torch.rand(samples, self.Y, self.X, device=self.device) * 0.2) - 0.1

        exponent = (alpha00[:, None, :, :] + alpha10[:, None, :, :] * self.ex[None, :, None, None] +
                    alpha01[:, None, :, :] * self.ey[None, :, None, None] +
                    alpha11[:, None, :, :] * self.ex_plus_ey[None, :, None, None])
        f = torch.exp(exponent)
        return f

if __name__ == '__main__':
    # Test the GeneratingData class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Uax = 0.0
    Uay = 0.0
    Qn = 9
    X = 3001  # Example values
    Y = 5  # Example values
    samples = 10 # number of samples
    
    model = GeneratingData(device, Uax, Uay, Qn, X, Y)
    
    output = model(samples)
    
    print("Output shape:", output.shape)
    print("Output example:")
    print(output[0, 0, :5, :5]) 