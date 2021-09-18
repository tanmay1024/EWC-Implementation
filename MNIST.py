import random
import torch
import torchvision.datasets as dsets


class MyMNIST(torch.utils.data.Dataset):

    def __init__(self, TargetLabels, root, train, transform, download):
        self.transform = transform
        self.download = download
        self.train = train
        self.dataset = dsets.MNIST(root='./data', train=True, transform=self.transform, download=self.download)
        self.datasetSize = 0
        self.indicies = []
        self.label = []
        print("dataset_len", self.dataset.__len__())
        print("dataset_lentype", type(self.dataset.__len__()))
        for p in range(self.dataset.__len__()):
            each_input, each_label = self.dataset.__getitem__(p)
            for n in range(len(TargetLabels)):
                if each_label == TargetLabels[n]:
                    self.indicies.append(p)
                    self.datasetSize += 1
        self.mydata = torch.tensor([self.datasetSize, 28, 28])
        self.mydata = torch.zeros(self.datasetSize, 28, 28)
        for p in range(self.datasetSize):
            each_input, each_label = self.dataset.__getitem__(self.indicies[p])
            self.mydata[p] = each_input
            self.label.append(each_label)
        self.data = self.mydata.numpy()

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]