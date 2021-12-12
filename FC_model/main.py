import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
import os
import matplotlib.pyplot as plt
import torchvision
import sys
from fc_model import FCModel, ConvNet
import torch.utils

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_test_val_acc_loss_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IS_CNN = False
dim_in_feature = 32 * 10

def train(epoch_num, model, train_loader, optimizer):
    print('Epoch {}:'.format(epoch_num))
    model.train()
    train_loss = 0.0
    correct = 0.0
    for data, labels in train_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, dim_in_feature).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)

        train_loss += F.nll_loss(output, labels, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
    train_acc = float(100.0 * correct) / len(train_loader.sampler)
    train_loss = float(train_loss / len(train_loader.sampler))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.sampler), train_acc))
    return train_acc, train_loss


def validate(model, test_loader):
    model.eval()
    val_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, dim_in_feature).to(device)
        target = target.to(device)

        output = model(data)
        val_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
    val_loss /= len(test_loader.sampler)
    val_acc = float(100.0 * correct) / len(test_loader.sampler)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(test_loader.sampler), val_acc))
    return val_acc, val_loss

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, nums, dims):
        data_l = np.fromfile(data_root, sep="\t", dtype=np.float32)
        data_m = data_l.reshape(int(nums), dims + 1)
        self.label = data_m[:, -1]
        self.data = np.delete(data_m, -1, axis=1)
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, int(labels)
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# # 随机生成数据，大小为10 * 20列
# source_data = np.random.rand(10, 20)
# # 随机生成标签，大小为10 * 1列
# source_label = np.random.randint(0,2,(10, 1))
# # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
# torch_data = GetLoader(source_data, source_label)

def get_train_validation_test_loaders(data_dir, test_data_dir, batch_size, p_val, nums, dims):
    train_dataset = GetLoader(data_dir, nums, dims)
    valid_dataset = GetLoader(data_dir, nums, dims)
    test_dataset = GetLoader(test_data_dir, 1e3, dims)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(p_val * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False)

    return (train_loader, valid_loader, test_loader)

def get_target_loaders(data_dir, batch_size, p_val, p_test, nums, dims):
    train_dataset = GetLoader(data_dir, nums, dims)
    valid_dataset = GetLoader(data_dir, nums, dims)
    test_dataset = GetLoader(data_dir, nums, dims)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split_val = int(np.floor((p_val + p_test) * num_train))
    split_test = int(np.floor(p_test * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[split_val:], indices[split_test: split_val], indices[:split_test]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=1, pin_memory=True)

    return (train_loader, valid_loader, test_loader)


def save_test_predictions(model, path):
    test_x = np.loadtxt(path, delimiter="\t", dtype=np.float32)
    test_x = torch.from_numpy(test_x[:, 0:-1]).to(device)

    output = model(test_x)
    preds = output.data.max(1, keepdim=True)[1]

    with open('output/test.pred', 'w+') as f:
        f.writelines(map(lambda x: str(int(x)) + '\n', preds))


def create_confusion_matrix(model, test_loader, categories_num=10):
    confusion_martix = []
    for i in range(categories_num):
        confusion_martix.append([0] * categories_num)
    for data, labels in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, dim_in_feature).to(device)

        labels = labels.to(device)

        output = model(data)
        preds = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # correct += pred.eq(labels.data.view_as(pred)).sum()
        for pred, label in zip(preds, labels):
            confusion_martix[label][pred] += 1

    print('\nConfustion matrix:\n')
    for l in range(categories_num):
        print(confusion_martix[l])


def print_test_acc_loss(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if IS_CNN:
            data = data.to(device)
        else:
            data = data.reshape(-1, dim_in_feature).to(device)

        target = target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.sampler)
    test_acc = float(100.0 * correct) / len(test_loader.sampler)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.sampler), test_acc))
    return test_acc, test_loss


def main():
    print(sys.version)
    print(torch.__version__)
    print('GPU is available' if torch.cuda.is_available() else 'GPU is not available')

    train_data_dir = r'/home/ljun/anns/hnsw_nics/experiment/aware/static/target_300/d1_w10_r1/train_49.txt'
    test_data_dir = r'/home/ljun/anns/hnsw_nics/experiment/aware/static/target_300/d1_w10_r1/test_49.txt'

    train_loader, val_loader, test_loader = get_train_validation_test_loaders(train_data_dir, test_data_dir, 16, 0.1, 1e4, dim_in_feature)
    # train_loader, val_loader, test_loader = get_target_loaders(test_data_dir, 32, 0.2, 0.2, 1e3, dim_in_feature)

    if IS_CNN:
        model = ConvNet().to(device)
    else:
        model = FCModel(dim_in_feature, 48, 2).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
    for epoch in range(50):
        train_acc, train_loss = train(epoch + 1, model, train_loader, optimizer)
        val_acc, val_loss = validate(model, val_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

    print_test_acc_loss(model, test_loader)

    save_test_val_acc_loss_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    create_confusion_matrix(model, test_loader, 2)

    # save model parameters
    torch.save(model.state_dict(), 'mymodel.pkl')
    # load save model parameters
    # model_object.load_state_dict(torch.load('mymodel.pkl'))
    save_test_predictions(model, test_data_dir)


if __name__ == '__main__':
    main()
