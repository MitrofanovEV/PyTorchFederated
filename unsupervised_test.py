from federated import single
from datasets.FASDataset import FacesDataset
import torch
import torchvision
import os
from pathlib import Path

train_labeled = []
train_unlabeled = []

with open('train.txt', 'r') as file:
    for line in file.readlines():
        train_labeled.append(Path(line[0:-1]))

with open('train_unlabeled.txt', 'r') as file:
    for line in file.readlines():
        train_unlabeled.append(Path(line[0:-1]))

print(len(train_labeled), len(train_unlabeled))
print(train_unlabeled[0])
train_dataset = FacesDataset(train_labeled, mode='train', augmentations=True)

train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=64,
                                       pin_memory=True,
                                       shuffle=True)

path = os.getcwd()
fn = path + '/experiments/' + '/2_UNLABELED'
fplot = fn + '/plots'
try:
    os.mkdir(fn)
    os.mkdir(fplot)
except FileExistsError:
    pass
dir_val = Path(path + '/data/Faces/Evaluation_Data/')

val_files = sorted(list(dir_val.rglob('*.png')))
val_dataset = FacesDataset(val_files, mode='val', augmentations=False)
val_dl = torch.utils.data.DataLoader(dataset=val_dataset,
                                     batch_size=64,
                                     pin_memory=True,
                                     shuffle=False)

val_loss1, val_acc1, model_s = single(torchvision.models.resnet50, train_dl, val_dl, fn, lr=1e-5, n_epochs=10,
                                      epoch_step=1, num_classes=2)

torch.save(model_s.state_dict(), 'supervised_model')
