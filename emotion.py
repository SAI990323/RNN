import argparse

import load
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import net
from tensorboardX import SummaryWriter



class MyData(data.Dataset):
    def __init__(self, train=True, transform=None, transform_target=None):
        self.train = train
        if self.train:
            self.data, self.target,_,_ = load.get_data()
        else:
            _,_,self.data, self.target = load.get_data()
        self.target = torch.LongTensor(self.target)
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.transform_target is not None:
            target = self.transform_target(target)

        return data, target

    def __len__(self):
        return len(self.data)

device = torch.device("cuda")
transform = torchvision.transforms.Compose([transforms.ToTensor(),])
trainset = MyData(train=True, transform=transform)
testset = MyData(train=False, transform=transform)


class EmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmotionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = net.MyCell(input_size, hidden_size)
        self.lstm2 = net.MyCell(hidden_size,hidden_size)
        self.linear = nn.Linear(hidden_size * 50, 2)

    def forward(self, input):
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        output = None
        input = input.squeeze(1)
        for i, data in enumerate(input.chunk(input.size(1), dim=1)):
            data = data.to(device)
            h_t, c_t = self.lstm1(data.squeeze(1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            if output == None:
                output = h_t2
            else :
                output = torch.cat((output, h_t2), dim = 1)
        output = self.linear(output)
        return F.softmax(input=output, dim = 1)


def train(epoch, learning_rate, batch_sie, trainset, testset):
    trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_sie)
    writer = SummaryWriter(comment = 'Emotion')
    lossfunc = nn.CrossEntropyLoss()
    model = EmotionNet(50,100).to(device).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    for i in range(epoch):
        total_loss = 0
        correct = 0
        for data, target in trainset:
            inputs = data.to(device)
            targets = target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossfunc(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == targets).sum().item()
        print("epoch %d: train_acc: %.3f" % (i, correct / 8662))
        print("epoch %d: loss: %.3f" % (i, total_loss / len(trainset)))
        writer.add_scalar('Train', total_loss / len(trainset), i)
        test_acc = test(model, testset, batch_sie)
        writer.add_scalar('Test', test_acc, i)
        # learning_rate = learning_rate * 0.98
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    writer.close()

    torch.save(model, "Emotion.model")

def test(model, testset, batch_size):
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testset:
            inputs = data.to(device)
            targets = target.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total = total + len(inputs)
    print('test data accuracy: ', correct / total)
    return correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=128, type=int)
    parser.add_argument("--epoch", dest="epoch", default=100, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.01,type=float)
    args = parser.parse_args()
    train(epoch=args.epoch, learning_rate=args.lr, batch_sie=args.batch_size, trainset = trainset, testset = testset)
