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
    def __init__(self, hidden_size=400, learning='class_learning'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        if learning == 'task_learning':
            self.fc4 = nn.Linear(hidden_size, 2)
        else:
            self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, images):
        x = F.relu(self.fc1(images))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class main():
    def __init__(self, learning, epochs, lr, batch_size, sample_size, hidden_size, num_task, importance):
        self.learning = learning
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.num_task = num_task
        self.importance = importance
        use_cuda = True
        weight = None
        model = MLP(self.hidden_size, self.learning)
        train_loader = {}
        test_loader = {}
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)
        if self.learning == 'class_learning':
            test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
            print("Test Dataset Size: ", (test_dataset.__len__()))
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

        self.loss = {}
        self.ewc = {}
        if self.learning == 'task_learning':
            self.acc = np.zeros((self.num_task, 2))
        else:
            self.acc = np.zeros((self.num_task, 10))

        label_step = int(10 / self.num_task)
        for task in range(self.num_task):
            target_labels = []
            for label in range(task * label_step, (task + 1) * label_step):
                target_labels.append(label)
            train_dataset = MyMNIST(target_labels, './data', True, transforms.ToTensor(), True)
            print("Train Dataset Size:", (train_dataset.__len__()))
            if self.learning == 'task_learning':
                test_dataset = MyMNIST(target_labels, './data', False, transforms.ToTensor(), True)
                print("Test Dataset Size: ", (test_dataset.__len__()))
                test_loader[task] = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

            train_loader[task] = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                             shuffle=True)

            self.loss[task] = []
            # self.acc[task] = []

            if task == 0:
                if weight:
                    model.load_state_dict(weight)
                else:
                    for _ in tqdm(range(self.epochs)):
                        self.loss[task].append(normal_train(model, optimizer, train_loader[task]))
                    if self.learning == 'class_learning':
                        self.acc[task] = evaluation(model, test_loader, self.num_task, task=0,
                                                    learning='class_learning')  # , self.num_task
                    else:
                        self.acc[task] = evaluation(model, test_loader[task], self.num_task, task=0,
                                                    learning='task_learning')
            else:
                old_tasks = []
                for sub_task in range(task):
                    old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(self.sample_size)
                old_tasks = random.sample(old_tasks, k=self.sample_size)
                for _ in tqdm(range(self.epochs)):
                    self.loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks),
                                                     self.importance, task))
                if self.learning == 'class_learning':
                    self.acc[task] = evaluation(model, test_loader, self.num_task, task, learning='class_learning')
                else:
                    for sub_task in range(task + 1):
                        self.acc[sub_task] = evaluation(model, test_loader[sub_task], self.num_task, sub_task,
                                                        learning='task_learning')
            print("Accuracy Matrix:", self.acc)
            print("Average Accuracy:", np.sum(self.acc) / 2*(task+1))


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage python3.8 main.py [ype of Learning(task_learning/class_learning)][# of epochs][learning rate]"
              "[batch size][sample size][hidden size][# of tasks][importance]")
    else:
        MAIN = main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                    int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]))

# main('class_learning', 100, 0.001, 128, 200, 400, 5, 1000)
