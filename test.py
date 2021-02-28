import torch
import torchvision
import torchvision.transforms as transforms
import os
from federated import experiment

num_classes = 27
n_experiments = 10
batch_size = 32
lr = 1e-3
n_clients = 10
hidden_size = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset = torchvision.datasets.EMNIST(root='data/',
                                            split='letters',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_dataset = torchvision.datasets.EMNIST(root='data/',
                                           split='letters',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

experiment(torchvision.models.resnet50, train_dataset, test_dataset, prefix='EMNIST',
           n_epochs=5, n_clients=5, n_experiments=1, learning_rate=1e-3, epoch_step=1, batch_size=1024)
