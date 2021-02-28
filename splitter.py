import os
from pathlib import Path
from sklearn.model_selection import train_test_split

path = os.getcwd()
dir_train = Path(path + '/data/Faces/Training_Data/')

train_files = sorted(list(dir_train.rglob('*.png')))

labels = [int(path.parent.parent.name == 'real') for path in train_files]

train, val = train_test_split(train_files, test_size=0.6, stratify=labels)

with open('train.txt', 'w') as file:
    for img in train:
        file.write(str(img) + '\n')

with open('train_unlabeled.txt', 'w') as file:
    for img in val:
        file.write(str(img) + '\n')
