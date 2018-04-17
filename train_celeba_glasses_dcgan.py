from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
import numpy as np


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
        # labels = self.labels.iloc[idx, 1].astype('int')
        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = np.argmax(labels)
        if self.transform:
            image = self.transform(image)
        return [image, labels]


class FeatureExtractor(object):

    def __init__(self, attr_file_path, transform, batch_size=64):
        f = open(attr_file_path, 'r')
        self.lines = f.readlines()
        n_lines = self.lines[0]
        self.features = self.lines[1].split(' ')
        self.transform = transform
        self.batch_size = batch_size

    def extract_feature(self, feature):
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

        feature_id = self.features.index(feature)
        pos_ = []
        neg_ = []
        for line in self.lines[2:]:
            line = split_line(line)
            label = int(line[1][feature_id] == '1')
            img_path = line[0]
            l = [0, 1] if label == 1 else [1, 0]
            if label == 0:
                neg_.append(np.array([img_path, l[0], l[1]]))
            else:
                pos_.append(np.array([img_path, l[0], l[1]]))

        return np.array(pos_), np.array(neg_)

    def build_data_loader(self, images, imgs_directory):
        """
        Build a dataloader for training a GAN

        :param images:
            list of images created by extract_feature
        :return:
            dataloader for training a GAN
        """
        labels = pd.DataFrame({
            'img_name': images[:, 0],
            'pos': images[:, 1],
            'neg': images[:, 2]
        })
        ds = CustomDataset(labels, imgs_directory, transform=self.transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return dl



""" gpu """
gpu_id = [0]
utils.cuda_devices(gpu_id)


""" param """
epochs = 50
batch_size = 64
lr = 0.0002
z_dim = 100


""" data """
crop_size = 108
re_size = 64
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




feature_extractor = FeatureExtractor('data/list_attr_celeba.txt', transform=transform)
extracted_features = feature_extractor.extract_feature('Eyeglasses')
data_loader = feature_extractor.build_data_loader(extracted_features[0], '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/train/img_align_celeba')

# imagenet_data = dsets.ImageFolder('/home/ashbylepoc/PycharmProjects/gan-tutorial/data', transform=transform)
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          num_workers=4)


""" model """
D = models_64x64.Discriminator(3)
G = models_64x64.Generator(z_dim)
bce = nn.BCEWithLogitsLoss()
utils.cuda([D, G, bce])

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


""" load checkpoint """
ckpt_dir = './checkpoints/celeba_dcgan_1'
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


""" run """
writer = tensorboardX.SummaryWriter('./summaries/celeba_dcgan_glasses')

z_sample = Variable(torch.randn(100, z_dim))
z_sample = utils.cuda(z_sample)
for epoch in range(start_epoch, epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable(imgs)
        bs = imgs.size(0)
        z = Variable(torch.randn(bs, z_dim))
        r_lbl = Variable(torch.ones(bs))
        f_lbl = Variable(torch.zeros(bs))
        imgs, z, r_lbl, f_lbl = utils.cuda([imgs, z, r_lbl, f_lbl])

        f_imgs = G(z)

        # train D
        r_logit = D(imgs)
        f_logit = D(f_imgs.detach())
        d_r_loss = bce(r_logit, r_lbl)
        d_f_loss = bce(f_logit, f_lbl)
        d_loss = d_r_loss + d_f_loss

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalars('D',
                           {"d_loss": d_loss.data.cpu().numpy(),
                            "d_r_loss": d_r_loss.data.cpu().numpy(),
                            "d_f_loss": d_f_loss.data.cpu().numpy()},
                           global_step=step)

        # train G
        f_logit = D(f_imgs)
        g_loss = bce(f_logit, r_lbl)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        writer.add_scalars('G',
                           {"g_loss": g_loss.data.cpu().numpy()},
                           global_step=step)

        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(data_loader)))

        if (i + 1) % 100 == 0:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2.0

            save_dir = './sample_images_while_training/celeba_dcgan_glasses'
            utils.mkdir(save_dir)
            torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)

    utils.save_checkpoint({'epoch': epoch + 1,
                           'D': D.state_dict(),
                           'G': G.state_dict(),
                           'd_optimizer': d_optimizer.state_dict(),
                           'g_optimizer': g_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)

# SAMPLE FROM SAVED GAN
def sample(n_samples):
    z_sample = Variable(torch.randn(n_samples, z_dim))
    z_sample = utils.cuda(z_sample)
    imgs = (G(z_sample).data + 1) / 2.0
    save_dir = './sample_images/celeba_dcgan_glasses'
    utils.mkdir(save_dir)
    torchvision.utils.save_image(f_imgs_sample,
                                 '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)