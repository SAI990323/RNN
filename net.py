import torch
import torch.nn as nn

device = torch.device("cuda")

class MyCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = torch.nn.Linear(input_size , hidden_size, bias=True)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer2 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.layer6 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer3 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.layer7 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer4 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.layer8 = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input, hx = None):
        if hx == None:
            h = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double).to(device)
            c = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double).to(device)
        else:
            h = hx[0]
            c = hx[1]

        i = torch.sigmoid(self.layer1(input) + self.layer5(h))
        f = torch.sigmoid(self.layer2(input) + self.layer6(h))
        o = torch.sigmoid(self.layer3(input) + self.layer7(h))
        nc = torch.tanh(self.layer4(input) + self.layer8(h))
        ct = f * c + i * nc
        ht = o * torch.tanh(ct)
        return ht,ct