import sys
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from MNIST import MyMNIST
from ewc import EWC, ewc_train, normal_train, evaluation
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class MLP(nn.Module):
    def __init__(self, hidden_size=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, images):
        x = F.relu(self.fc1(images))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class main():
    def __init__(self, epochs, lr, batch_size, sample_size, hidden_size, num_task, importance):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.num_task = num_task
        self.importance = importance
        use_cuda = True
        weight = None
        model = MLP(self.hidden_size)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

        self.loss = {}
        self.acc = np.zeros((self.num_task, 10))
        self.ewc = {}
        train_loader = {}
        # test_loader = {}
        # permute_idx = 28*28
        label_step = int(10 / self.num_task)
        for task in range(self.num_task):
            target_labels = []
            for label in range(task * label_step, (task + 1) * label_step):
                target_labels.append(label)
            train_dataset = MyMNIST(target_labels, './data', True, transforms.ToTensor(), True)
            # test_dataset = MyMNIST(target_labels, './data', False, transforms.ToTensor(), True)
            print("main.__init__() datasetsize=%s" % (train_dataset.__len__()))
            # print(img.shape for ())
            # train_dataset = torch.stack([img.float().view(-1)[permute_idx-1] / 255 for (img, __) in train_dataset])
            train_loader[task] = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                             shuffle=True)
            # test_loader[task] = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)
            self.loss[task] = []
            #self.acc[task] = []

            if task == 0:
                if weight:
                    model.load_state_dict(weight)
                else:
                    for _ in tqdm(range(self.epochs)):
                        self.loss[task].append(normal_train(model, optimizer, train_loader[task]))
                    self.acc[task] = evaluation(model, test_loader, self.num_task, task=0)  # , self.num_task
            else:
                old_tasks = []
                for sub_task in range(task):
                    old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(self.sample_size)
                old_tasks = random.sample(old_tasks, k=self.sample_size)
                for _ in tqdm(range(self.epochs)):
                    self.loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks),
                                                     self.importance))
                self.acc[task] = evaluation(model, test_loader, self.num_task, task) # , self.num_task
            print("Accuracy Matrix:", self.acc)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage python3.8 main.py [# of epochs][learning rate][batch size][sample size]"
              "[hidden size][# of tasks][importance]")
    else:
        MAIN = main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                    int(sys.argv[6]), float(sys.argv[7]))

# main(100, 0.001, 128, 200, 400, 5, 1000)
