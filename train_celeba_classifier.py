import time
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
squeezenet = models.squeezenet1_1(pretrained=True)

# load inputs + labels
""" data """
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

valid_data = dsets.ImageFolder('/home/ashbylepoc/PycharmProjects/gan-tutorial/data/valid/', transform=transform)
train_data = dsets.ImageFolder('/home/ashbylepoc/PycharmProjects/gan-tutorial/data/train/', transform=transform)
data_loader_train = torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

data_loader_valid = torch.utils.data.DataLoader(valid_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

#_, (inputs, _) = next(enumerate(data_loader_train))
#i = 0
#f = open('data_annotation_eyeglasses.csv', 'r').readlines()
#batch_lines = f[batch_size * i: batch_size * (i + 1)]
#labels = torch.FloatTensor([int(l[1]) for l in batch_lines])
#resnet = resnet.cuda()
#inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#outputs = resnet(inputs)
#outputs = squeezenet(inputs)

f = open('data_annotation_eyeglasses.csv', 'r').readlines()
labels_train = f[:170000]
labels_valid = f[170000:]

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

            for i, (inputs, _) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    lines = labels_train[i * batch_size: (i + 1) * batch_size]
                else:
                    lines = labels_valid[i * batch_size: (i + 1) * batch_size]
                # import pdb; pdb.set_trace()
                labels = torch.LongTensor([int(l[1]) for l in lines])
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
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

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
resnet.fc = torch.nn.Linear(num_ftrs, 16)

if use_gpu:
    resnet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dloaders = {'train': data_loader_train, 'valid': data_loader_valid}



if __name__ == '__main__':
    start_time = time.time()
    model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=2)
    print('Training time: {:10f} minutes'.format((time.time() - start_time)/60))

