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
import utils

z_dim = 100

G = models_64x64.Generator(z_dim)
utils.cuda([G])

""" load checkpoint """
ckpt_dir = './checkpoints/celeba_dcgan'
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    G.load_state_dict(ckpt['G'])
except:
    print(' [*] No checkpoint!')

def sample(n_samples):
    z_sample = Variable(torch.randn(n_samples, z_dim))
    z_sample = utils.cuda(z_sample)
    imgs = (G(z_sample).data + 1) / 2.0
    save_dir = './sample_images/celeba_dcgan'
    utils.mkdir(save_dir)
    return imgs

def birthday_paradox(num_batches=10, batch_size=100):
  save_dir = './sample_images/celeba_dcgan'
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


birthday_paradox(10, 100)
