from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for index, images in enumerate(self.dataset):
            images = torch.tensor(images)
            images = images.view(-1, 28*28)
            images = images.to(device)
            self.model.zero_grad()
            images = variable(images)
            output = self.model(images).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    epoch_loss = 0
    for index, (images, target) in enumerate(data_loader):
        images = images.view(-1, 28*28)
        images = images.to(device)
        target = target.to(device)
        # print(images)
        # print(target.shape)
        images, target = variable(images.clone().detach()), variable(target.clone().detach())
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, target)
        # print(loss)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    print("Loss: ", (epoch_loss / len(data_loader)))
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    for index, (images, target) in enumerate(data_loader):
        images = images.view(-1, 28 * 28)
        images = images.to(device)
        target = target.to(device)
        images, target = variable(images.clone().detach()), variable(target.clone().detach())
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    print("Loss: ", (epoch_loss / len(data_loader)))
    return epoch_loss / len(data_loader)

# Original evaluation method in the github repository(Not Used).


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for index, (images, target) in enumerate(data_loader):
        images = images.view(-1, 28*28)
        images, target = variable(images), variable(target)
        output = model(images)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    # print("Num of Correctly identified:", correct)
    # print("Dataset Length:", len(data_loader.dataset))
    return correct / len(data_loader.dataset)


def evaluation(model: nn.Module, data_loader, num_sessions, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # each_acc = torch.zeros([num_sessions, 10])
    each_acc = np.zeros((num_sessions, 10))
    div_acc = np.zeros((1, 10))
    # div_acc = div_acc.astype(np.float)
    # class_size = torch.zeros([10])
    # print("main.evaluation()  evaluation for this session")
    for index, (images, labels) in enumerate(data_loader):
        images = images.view(-1, 28 * 28)
        images = images.to(device)
        x = model(images)
        actualLabels = getLabel(x)
        # print("Output Labels: ", actualLabels[:20])
        # print("Actual Labels: ", labels[:20])
        labels = labels.to(device)
        for j in range(10):
            correct = 0
            for i in range(len(actualLabels)):
                if actualLabels[i] == labels[i] & labels[i] == j:
                    correct += 1
            div_acc[0][j] = correct
        each_acc[task] = div_acc
        # for c in range(10):
        #    div_acc[0][c] = ((actualLabels == labels) * (labels == c)).sum() / (labels == c).sum()
        # each_acc[task] = div_acc
    return each_acc[task]


def getLabel(outputs):
    return outputs.max(-1)[1]
