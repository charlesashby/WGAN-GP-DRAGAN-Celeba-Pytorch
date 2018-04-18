from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision import models
import utils
import sys

z_dim = 100

G = models_64x64.Generator(z_dim)
resnet = models.resnet50()
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2)
utils.cuda([G, resnet])

""" load checkpoint """
try:
    ckpt_dir = './checkpoints/celeba_dcgan'
    ckpt = utils.load_checkpoint(ckpt_dir)
    G.load_state_dict(ckpt['G'])

    ckpt_dir = './checkpoints/celeba_dcgan_blond'
    ckpt = utils.load_checkpoint(ckpt_dir)
    resnet.load_state_dict(ckpt['model'])
except:
    print(' [*] No checkpoint!')

G.eval()
resnet.eval()


def sample(n_samples):
    z_sample = Variable(torch.randn(n_samples, z_dim), volatile=True)
    z_sample = utils.cuda(z_sample)
    imgs = (G(z_sample).data + 1) / 2.0
    return imgs

def birthday_paradox(num_batches=10, batch_size=100):
  save_dir = './sample_images/celeba_dcgan_blond'
  utils.mkdir(save_dir)

  for n in range(num_batches):
    samples = sample(batch_size)
    closest = []
    for i in range(batch_size):
      closest.append((999999, None, i))
      for j in range(batch_size):
        if i != j:
          distance = torch.dist(samples[i], samples[j], 2)
          if distance < closest[i][0]:
            closest[i] = (distance, j, i)

    closest.sort()
    closest_unique = []
    for distance, i, j in closest:
      pair, inversed_pair = (i, j), (j, i)
      if pair not in closest_unique and inversed_pair not in closest_unique:
        closest_unique.append(pair)
        if len(closest_unique) == 10:
          break

    imgs = torch.zeros((20, 3, 64, 64))
    for i, pair in enumerate(closest_unique):
      imgs[2*i] = samples[pair[0]]
      imgs[(2*i)+1] = samples[pair[1]]

    torchvision.utils.save_image(imgs, '{}/batch_{}.jpg'.format(save_dir, n), nrow=2)

def birthday_paradox_grouping(num_batches=10, batch_size=100):
  save_dir = './sample_images/celeba_dcgan_blond'
  utils.mkdir(save_dir)

  for n in range(num_batches):
    samples = sample(batch_size).cpu()

    transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Scale(size=(224, 224), interpolation=Image.BICUBIC),
     transforms.ToTensor()])

    resized_samples = torch.zeros(batch_size, 3, 224, 224)
    for i, s in enumerate(samples):
      resized_samples[i] = transform(s)
    resized_samples = Variable(resized_samples.cuda(), volatile=True)
    outputs = resnet(resized_samples)
    _, preds = torch.max(outputs.data, 1)

    grouped_samples = {0: [], 1: []}
    for i, group in enumerate(preds):
      grouped_samples[group].append(samples[i])

    for k, v in grouped_samples.items():
      v = torch.stack(v)

      torchvision.utils.save_image(v, '{}/batch{}_all{}.jpg'.format(save_dir, n, k), nrow=10)

      closest = []
      for i in range(v.shape[0]):
        closest.append((999999, None, i))
        for j in range(v.shape[0]):
          if i != j:
            distance = torch.dist(v[i], v[j], 2)
            if distance < closest[i][0]:
              closest[i] = (distance, j, i)

      closest.sort()
      closest_unique = []
      for distance, i, j in closest:
        pair, inversed_pair = (i, j), (j, i)
        if pair not in closest_unique and inversed_pair not in closest_unique:
          closest_unique.append(pair)
          if len(closest_unique) == 10:
            break

      imgs = torch.zeros((20, 3, 64, 64))
      for i, pair in enumerate(closest_unique):
        imgs[2*i] = v[pair[0]]
        imgs[(2*i)+1] = v[pair[1]]

      torchvision.utils.save_image(imgs, '{}/batch{}_group{}.jpg'.format(save_dir, n, k), nrow=2)


birthday_paradox_grouping(10, 100)
