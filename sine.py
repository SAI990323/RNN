import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import net
device = torch.device("cuda")

class SineLSTM(nn.Module):
    def __init__(self):
        super(SineLSTM, self).__init__()
        self.lstm1 = net.MyCell(1, 50)
        self.lstm2 = net.MyCell(50, 50)
        self.linear = torch.nn.Linear(50, 1)

    def forward(self, input, predict=0):
        h_t = torch.zeros(input.size(0), 50, dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), 50, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), 50, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), 50, dtype=torch.double).to(device)
        input = input.chunk(input.size(1), dim = 1)
        outputs = []
        for i, input_t in enumerate(input):
            input_t = input_t.to(device)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(predict):
            output = output.to(device)
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


# 画线
def draw(y, test, i):
    plt.figure(figsize = (40,20))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    plt.plot(np.arange(999), test[0][:999], linewidth=2.0, color = 'blue')
    plt.plot(np.arange(999), test[1][:999], linewidth=2.0, color = 'yellow')
    plt.plot(np.arange(999), test[2][:999], linewidth=2.0, color = 'red')
    plt.plot(np.arange(999, 1999), y[0][999:], linestyle=':', linewidth=2.0, color = 'blue')
    plt.plot(np.arange(999, 1999), y[1][999:], linestyle=':', linewidth=2.0, color = 'yellow')
    plt.plot(np.arange(999, 1999), y[2][999:], linestyle=':', linewidth=2.0, color = 'red')
    print(i)
    plt.savefig('predict' + str(i) + '.jpg' )
    plt.close()


# 反向传播，利用 LBFGS 优化器进行优化
def closure():
    optimizer.zero_grad()
    out = net(input)
    loss = lossfunc(out, target).to(device)
    print('loss:', loss.item())
    loss.backward()
    return loss


def test(id):
    with torch.no_grad():
        predict = 1000
        pred = net(test_input, predict=predict)
        loss = lossfunc(pred[:, :-predict], test_target)
        print('test loss:', loss.item())
        y = pred.detach().cpu().numpy()
    draw(y, test_target.cpu().numpy(), id)


def train():
    for i in range(20):
        optimizer.step(closure)
        test(i)


if __name__ == '__main__':
    data = torch.load('./traindata.pt')
    input = torch.from_numpy(data[3:, :-1]).to(device)
    target = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)
    net = SineLSTM().to(device).double()
    lossfunc = nn.MSELoss()
    optimizer = optim.LBFGS(net.parameters(), lr=0.1)
    train()