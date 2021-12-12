import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out)


class FCModel(nn.Module):
    def __init__(self, in_dim, hidden_dim1, out_dim):
        super(FCModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, out_dim)
        # self.linear3 = nn.Linear(hidden_dim2, out_dim)
        # self.batch_norm1 = nn.BatchNorm1d(hidden_dim1)
        # self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)

    # # #with batchnorm and 
    # def forward(self, input):
    #     out = F.tanh(self.batch_norm1(self.linear1(input)))
    #     out = F.tanh(self.batch_norm2(self.linear2(out)))
    #     out = self.linear3(out)
    #     return F.log_softmax(out)

    # # #with batchnorm
    # def forward(self, input):
    #     out = F.relu(self.batch_norm1(self.linear1(input)))
    #     out = F.relu(self.batch_norm2(self.linear2(out)))
    #     out = self.linear3(out)
    #     return F.log_softmax(out)

    # # # with dropout
    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.dropout(out, p=0.1, training=self.training)
        # out = F.relu(self.linear2(out))
        # out = F.dropout(out, p=0.1, training=self.training)
        out = self.linear2(out)
        return F.log_softmax(out)


    # #regular
    # def forward(self, input):
    #     out = F.relu(self.linear1(input))
    #     out = F.relu(self.linear2(out))
    #     out = self.linear3(out)
    #     return F.log_softmax(out)