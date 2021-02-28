import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets.FASDataset import FacesDataset
from federated import experiment
import torchvision
import torch
path = os.getcwd()
dir_train = Path(path + '/data/Faces/Training_Data/')

train_files = sorted(list(dir_train.rglob('*.png')))
print(train_files[0])
dir_val = Path(path + '/data/Faces/Evaluation_Data/')

val_files = sorted(list(dir_val.rglob('*.png')))

#labels = [int(path.parent.parent.name == 'real') for path in train_files]

train_dataset = FacesDataset(train_files, mode='train', augmentations=True)
val_dataset = FacesDataset(val_files, mode='val', augmentations=False)
img, label = val_dataset[0]
print(torch.max(img))
print(torch.min(img))
experiment(torchvision.models.resnet50, train_dataset, val_dataset, prefix='FAS_resnet50_With_Crop_TT',
           n_epochs=15, n_clients=5, n_experiments=1, learning_rate=1e-5, epoch_step=1, batch_size=64)


