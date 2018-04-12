import time
import pandas as pd
import random, csv
import numpy as np
import datetime as dt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as dsets
from torchvision import transforms, datasets, models

use_gpu = torch.cuda.is_available()
resnet = models.resnet50(pretrained=True)
# squeezenet = models.squeezenet1_1(pretrained=True)

# ----------------------------------------------------------------------
#                     HOW TO USE THE MODEL
# ----------------------------------------------------------------------
#
# 1. Create a .csv file containing tuples ('img_name.jpg', label==[{0,1}, {0,1}])
# 2. Load this csv in labels
# 3. Change the path to the images in train_ds and valid_ds
# 4. Run it
# 5. My results: (Eyeglasses)
# >>> Epoch [1/1] train loss: 0.0026 acc: 0.9396 valid loss: 0.0023 acc: 0.9481
# >>> Best val Acc: 0.948052
# >>> Training time:   1.691098 minutes
# ----------------------------------------------------------------------

# I CHOOSE THE FEATURE YOU WANT TO EXTRACT + CHECK DATA DISTRIBUTION


def split_line(line):
    data = []
    line = line.split(' ')
    for l in line:
        if l != '':
            data.append(l)
    return data[0], data[1:]


f = open('data/list_attr_celeba.txt', 'r')
lines = f.readlines()
n_lines = lines[0]
features = lines[1].split(' ')
feature = 'Eyeglasses'
feature_id = features.index(feature)

pos_ = []
neg_ = []
for line in lines[2:]:
    line = split_line(line)
    label = int(line[1][feature_id] == '1')
    img_path = line[0]
    l = [0, 1] if label == 1 else [1, 0]
    if label == 0:
        pos_.append((img_path, l[0], l[1]))
    else:
        neg_.append((img_path, l[0], l[1]))


# II REBALANCE DATA SET BECAUSE ONLY < 5% OF IMAGES HAVE GLASSES
data = neg_ + pos_[:len(neg_)]
random.Random(1).shuffle(data)

# III CREATE CSV FILES WITH THE NAME OF THE IMAGES + TARGETS
file_name = 'data/labels.csv'
with open(file_name,'wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['img_name', 'with_glasses', 'without_glasses'])
    for row in data:
        csv_out.writerow(row)


# IV LOAD IMAGES, NORMALIZE, CROP, ETC

batch_size = 64
crop_size = 108
re_size = 224
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(crop),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
# data = dsets.ImageFolder('/home/ashbylepoc/PycharmProjects/gan-tutorial/data/', transform=transform)


# V MERGE IMAGES AND TARGETS FOR DATA LOADER
labels = pd.read_csv('data/labels.csv')
labels_train = labels[:25000]
labels_valid = labels[25000:]


class CustomDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = '{}'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = np.argmax(labels)
        if self.transform:
            image = self.transform(image)
        return [image, labels]


train_ds = CustomDataset(labels_train, '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/glasses/train/train/',
                         transform=transform)
valid_ds = CustomDataset(labels_valid, '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/glasses/valid/valid/',
                         transform=transform)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=True, num_workers=4)

# VI TEST IT
# img, label = next(iter(train_dl))


# VII TRAIN THE MODEL

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'valid': len(dataloaders['valid'].dataset)}

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # if phase == 'train':
                #     lines = labels_train[i * batch_size: (i + 1) * batch_size]
                # else:
                #     lines = labels_valid[i * batch_size: (i + 1) * batch_size]
                # import pdb; pdb.set_trace()
                # labels = torch.LongTensor([int(l[1]) for l in lines])
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                corrects = torch.sum(preds == labels.data)
                running_corrects += corrects

                print('Epoch [{}/{}] Iteration [{}/{}] loss: {:.4f} acc: {:.4f}'.format(
                    epoch, num_epochs-1, i, len(dataloaders[phase]),
                    loss.data[0], float(corrects) / float(len(labels.data))))

            if phase == 'train':
                train_epoch_loss = float(running_loss) / float(dataset_sizes[phase])
                train_epoch_acc = float(running_corrects) / float(dataset_sizes[phase])
            else:
                valid_epoch_loss = float(running_loss) / float(dataset_sizes[phase])
                valid_epoch_acc = float(running_corrects) / float(dataset_sizes[phase])

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
            epoch, num_epochs - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


for param in resnet.parameters():
    param.requires_grad = False

num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2)

if use_gpu:
    resnet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
dloaders = {'train': train_dl, 'valid': valid_dl}


if __name__ == '__main__':
    start_time = time.time()
    model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=2)
    print('Training time: {:10f} minutes'.format((time.time() - start_time)/60))

