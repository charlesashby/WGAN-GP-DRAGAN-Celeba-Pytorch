import time
import pandas as pd
import random, csv
import numpy as np
import datetime as dt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
import utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as dsets
import resnet50
from torchvision import transforms, datasets, models
import torch.nn.functional as F

DATASET = '/Tmp/brouilph/datasets/img_align_celeba/'
ANNOTATION = '/Tmp/brouilph/datasets/img_align_celeba/data/list_attr_celeba.txt'
OUTPUT_IMG = ''
OUTPUT_CHECKPOINT = ''

use_gpu = torch.cuda.is_available()
resnet = resnet50.resnet50(pretrained=True)
resnet.cuda()
# squeezenet = models.squeezenet1_1(pretrained=True)

# ----------------------------------------------------------------------
#                     HOW TO USE THE MODEL
# ----------------------------------------------------------------------
#
# 1. Create a .csv file containing tuples ('img_name.jpg', label==[{0,1}, {0,1}])
# 2. Load that csv in "labels"
# 3. Change the path to the images in "train_ds" and "valid_ds"
# 4. Run it
# 5. My results: (Eyeglasses)
# >>> Epoch [1/1] train loss: 0.0026 acc: 0.9396 valid loss: 0.0023 acc: 0.9481
# >>> Best val Acc: 0.948052
# >>> Training time:   1.691098 minutes
# ----------------------------------------------------------------------

# IV LOAD IMAGES, NORMALIZE, CROP, ETC

batch_size = 64
crop_size = 108
re_size = 224
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
#crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
# data = dsets.ImageFolder('/home/ashbylepoc/PycharmProjects/gan-tutorial/data/', transform=transform)


# V MERGE IMAGES AND TARGETS FOR DATA LOADER
# labels = pd.read_csv('data/labels.csv')
# labels_train = labels[:25000]
# labels_valid = labels[25000:]

class FeatureExtractor(object):
    def __init__(self, attr_file_path, transform, batch_size=64):
        f = open(attr_file_path, 'r')
        self.lines = f.readlines()
        n_lines = self.lines[0]
        self.features = self.lines[1].split(' ')
        self.transform = transform
        self.batch_size = batch_size

    def get_nb_feature(self):
        return len(self.features) - 1

    def extract_feature(self, feature=None):
        """
        :param feature:
            Feature you want to extract
        :return:
            Returns 2 lists with the names of
            every images with the feature (pos_)
            and who don't have the feature (neg_)
        """

        def split_line(line):
            data = []
            line = line.split(' ')
            for l in line:
                if l != '':
                    data.append(l)
            return data[0], data[1:]

        #feature_id = self.features.index(feature)
        result = []
        img_paths = []
        for line in self.lines[2:]:
            line = split_line(line)
            img_path = line[0]
            img_paths.append(img_path)
            label = map(int, line[1])
            label = np.array( label)
            label = (label + 1) / 2

            result.append(label)
        return img_paths, np.array(result)

    def build_data_loader(self, img_paths, results, imgs_directory, train=True):
        """
        Build a dataloader for training a GAN

        :param images:
            list of images created by extract_feature
        :return:
            dataloader for training a GAN
        """
        #labels = pd.DataFrame({
        #    'img_name': img_paths,
        #    'labels': results,
        #})
        if train:
            img_paths = img_paths[:int(len(img_paths) * 0.80)]
            results = results[:int(len(results) * 0.80)]
        else:
            img_paths = img_paths[int(len(img_paths) * 0.80):]
            results = results[int(len(results) * 0.80):]
        ds = CustomDataset(img_paths, results, imgs_directory, transform=self.transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return dl

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.img_paths = img_paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = '{}'.format(self.img_paths[idx])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return [image, labels]


feature_extractor = FeatureExtractor(ANNOTATION, transform=transform)
img_paths, results = feature_extractor.extract_feature()
train_dl = feature_extractor.build_data_loader(img_paths, results, DATASET, train=True)
valid_dl = feature_extractor.build_data_loader(img_paths, results, DATASET, train=False)

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
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                optimizer.zero_grad()

                outputs, hidden = model(inputs)
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


num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, feature_extractor.get_nb_feature())

if use_gpu:
    resnet.cuda()

def cross_entropy(x, y):
    import pdb;pdb.set_trace()
    loss = []
    x = F.log_softmax(x, 1)
    for i in range(x.size(1)):
        loss.append(F.binary_cross_entropy(x[:,i], y[:,i]))
    return sum(loss)

criterion = cross_entropy
#criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
dloaders = {'train': train_dl, 'valid': valid_dl}

if __name__ == '__main__':
    start_time = time.time()
    model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=2)
    ckpt_dir = OUTPUT_CHECKPOINT
    print('Training time: {:10f} minutes'.format((time.time() - start_time)/60))
    utils.save_checkpoint({'model': model.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, 1))
    print('Saved model.')
