import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch

augmentations_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        #A.RandomSizedCrop((150, 224), 224, 224, p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

augmentations_pipeline1 = transforms.Compose(
    [
        #transforms.Resize((300, 300)),
        #transforms.RandomCrop(224),
        transforms.RandomResizedCrop(224, scale=(0.08, 1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomAffine(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)


class FacesDataset(Dataset):

    def __init__(self, files, mode, augmentations=False):
        super().__init__()
        self.files = files
        self.mode = mode
        self.augmentations = augmentations

        self.len_ = len(self.files)

        if self.mode == 'train':
            self.labels = [int(path.parent.parent.name == 'real') for path in self.files]
        if self.mode == 'val':
            labels = pd.read_csv('protocol_eval.txt', sep=' ')
            labels['File'] = labels['File'].apply(lambda x: x[-11:])
            self.labels = [int(labels.loc[labels['File'] == path.parent.name]['Label'] == 'real') for path in
                           self.files]

    def __len__(self):
        return self.len_

    def __getitem__(self, item):
        arr = Image.open(self.files[item]).convert("RGB")
        arr.load()
        #arr = arr.resize((224, 224))
        #arr = np.array(arr, dtype=np.float32)
        #arr /= 255.
        if self.augmentations:
            arr = augmentations_pipeline1(arr)
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            arr = transform(arr)
        if self.mode == 'test':
            return arr
        elif self.mode == 'train_unlabeled' or self.mode == 'val_unlabeled':
            label = random.randint(0, 3)
            for i in range(label):
                #print(arr.shape)
                arr = torch.rot90(arr, dims=(1, 2))
            return arr, label
        else:
            label = self.labels[item]
            return arr, label
