# coding: utf-8

# ----------------------------------------------------
# TO DO THE BP PARADOX TEST:
#  python main.py --cuda --bp 1 --train 0 --b_size 64
# b_size is the number of images to generate
# -----------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque
import torchvision.utils as vutils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
import os
import os.path
import torchvision
import PIL.Image as Image
from dataloader import *
from models import *
import argparse
import random

experiment_path = 'experiments'


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--bp', default=0, type=int)
parser.add_argument('--b_size', default=16, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_update_step', default=3000, type=int)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--lr_lower_boundary', default=2e-6, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--lambda_k', default=0.001, type=float)
parser.add_argument('--k', default=0, type=float)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--model_name', default='test2')
parser.add_argument('--base_path', default='/home/ashbylepoc/PycharmProjects/WGAN-GP-DRAGAN-Celeba-Pytorch/began')
parser.add_argument('--data_path', default='/home/ashbylepoc/PycharmProjects/gan-tutorial/data/train/img_align_celeba')
parser.add_argument('--load_step', default=0, type=int)
parser.add_argument('--print_step', default=100, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--l_type', default=1, type=int)
parser.add_argument('--tanh', default=1, type=int)
parser.add_argument('--manualSeed', default=5451, type=int)
parser.add_argument('--train', default=1, type=int)
opt = parser.parse_args()


print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    opt.cuda = True
    torch.cuda.set_device(opt.gpuid)
    torch.cuda.manual_seed_all(opt.manualSeed)


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

    def __init__(self, attr_file_path, transform, batch_size=opt.b_size):
        f = open(attr_file_path, 'r')
        self.lines = f.readlines()
        n_lines = self.lines[0]
        self.features = self.lines[1].split(' ')
        print(self.features)
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


""" data """
crop_size = 108
re_size = 64
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
# crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose(
    [transforms.ToTensor(),
#    transforms.Lambda(crop),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

feature_extractor = FeatureExtractor('/home/ashbylepoc/PycharmProjects/GAN-pytorch/data/list_attr_celeba.txt', transform=transform)
extracted_features = feature_extractor.extract_feature('No_Beard')
data_loader = feature_extractor.build_data_loader(extracted_features[1],
                                '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/train/img_align_celeba')

class BEGAN():
    def __init__(self):
        self.global_step = opt.load_step
        self.prepare_paths()
        # self.data_loader = get_loader(self.data_path, opt.b_size, opt.scale_size, opt.num_workers)
        self.data_loader = data_loader
        self.build_model()

        self.z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z.data.uniform_(-1, 1)
        self.fixed_x = None
        self.criterion = L1Loss()

        if opt.cuda:
            self.set_cuda()

    def set_cuda(self):
        self.disc.cuda()
        self.gen.cuda()
        self.z = self.z.cuda()
        self.fixed_z = self.fixed_z.cuda()
        self.criterion.cuda()

    def write_config(self, step):
        f = open(os.path.join(opt.base_path,
                              experiment_path + '/%s/params/%d.cfg' % (opt.model_name, step)), 'w')
        print(f, vars(opt))
        f.close()
 
    def prepare_paths(self):
        utils.mkdir(experiment_path)
        self.data_path = opt.data_path
        self.gen_save_path = os.path.join(opt.base_path, experiment_path + '/%s/models' % opt.model_name)
        self.disc_save_path = os.path.join(opt.base_path, experiment_path + '/%s/models' % opt.model_name)
        self.sample_dir = os.path.join(opt.base_path,  experiment_path + '/%s/samples' % opt.model_name)
        param_dir =  os.path.join(opt.base_path,  experiment_path + '/%s/params' % opt.model_name)

        for path in [self.gen_save_path, self.disc_save_path, self.sample_dir, param_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        print("Generated samples saved in %s" % self.sample_dir)
    
    def build_model(self):
        self.disc = Discriminator(opt)
        self.gen = Decoder(opt)
        print(self.disc)
        print("====================")
        print(self.gen)
        #disc.apply(weights_init)
        #gen.apply(weights_init)

        if opt.load_step > 0:
            self.load_models(opt.load_step)

    def generate(self, sample, recon, step, nrow=8):
        #sample = self.gen(fake)
        #print sample.size()
        #return
        #sample = sample.data.cpu().mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        #from PIL import Image
        #print type(sample)
        #im = Image.fromarray(sample.astype('uint8'))
        #im.save('128.png')

        vutils.save_image(sample.data,
                          '%s/%s_%s_gen.png' % (self.sample_dir, opt.model_name, str(step)),
                          nrow=nrow,
                          normalize=True)

        #f = open('%s/%s_gen.mat'%(self.sample_dir, opt.model_name), 'w')
        #np.save(f, sample.data.cpu().numpy())
        #recon = self.disc(self.fixed_x)

        if recon is not None:
            vutils.save_image(recon.data,
                              '%s/%s_%s_disc.png' % (self.sample_dir, opt.model_name, str(step)),
                              nrow=nrow,
                              normalize=True)

    def save_models(self, step):
        torch.save(self.gen.state_dict(), os.path.join(self.gen_save_path, 'gen_%d.pth'%step)) 
        torch.save(self.disc.state_dict(), os.path.join(self.disc_save_path, 'disc_%d.pth'%step)) 
        self.write_config(step)

    def load_models(self, step):
        # import pdb; pdb.set_trace()
        self.gen.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'gen_%d.pth'%step)))     
        self.disc.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'disc_%d.pth'%step)))     

    def compute_disc_loss(self, outputs_d_x, data, outputs_d_z, gen_z):
        if opt.l_type == 1:
            real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
            fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
        else:
            real_loss_d = self.criterion(outputs_d_x, data)
            fake_loss_d = self.criterion(outputs_d_z , gen_z.detach())
        return (real_loss_d, fake_loss_d)
            
    def compute_gen_loss(self, outputs_g_z, gen_z):
        if opt.l_type == 1:
            return torch.mean(torch.abs(outputs_g_z - gen_z))
        else:
            return self.criterion(outputs_g_z, gen_z)

    def sample(self, n_samples):
        # z_dim = opt.h
        self.z.data.uniform_(-1, 1)
        # import pdb; pdb.set_trace()
        # z_sample = Variable(torch.randn(n_samples, z_dim), volatile=True)
        # z_sample = utils.cuda(z_sample)
        imgs = (self.gen(self.z).data + 1) / 2.0
        return imgs

    def birthday_paradox(self, num_batches=10, batch_size=opt.b_size):

        # load ckpt
        try:
            print('loading model')
            self.load_models(136000)
            print('model loaded')
        except:
            pass

        save_dir = './sample_images/{}'.format(experiment_path)
        utils.mkdir(save_dir)
        for n in range(num_batches):
            samples = self.sample(batch_size)
            closest = []
            for i in range(batch_size - 1):
                closest.append((999999, None, i))
                for j in range(batch_size - 1):
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
                imgs[2 * i] = samples[pair[0]]
                imgs[(2 * i) + 1] = samples[pair[1]]

            torchvision.utils.save_image(imgs, '{}/batch_{}.jpg'.format(save_dir, n), nrow=2)

    def train(self):
        g_opti = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        d_opti = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        measure_history = deque([0] * opt.lr_update_step, opt.lr_update_step)
        convergence_history = []
        prev_measure = 1
        lr = opt.lr

        # import pdb; pdb.set_trace()
        for i in range(opt.epochs):
            for _, (data, _) in enumerate(self.data_loader):
                # print(self.global_step)
                data = Variable(data)
                if data.size(0) != opt.b_size:
                    print(data.size(0))
                    print(opt.b_size)
                    continue

                if opt.cuda:
                    data = data.cuda()
                if self.fixed_x is None:
                    self.fixed_x = data
                #self.gen.zero_grad()
                self.disc.zero_grad()

                self.z.data.uniform_(-1, 1)
                gen_z = self.gen(self.z)
                outputs_d_z = self.disc(gen_z.detach())
                outputs_d_x = self.disc(data)
               
                real_loss_d, fake_loss_d = self.compute_disc_loss(outputs_d_x,
                                                                  data,
                                                                  outputs_d_z,
                                                                  gen_z)

                lossD = real_loss_d - opt.k * fake_loss_d
                lossD.backward()
                d_opti.step()
            
                self.gen.zero_grad()
                #self.disc.zero_grad()
                gen_z = self.gen(self.z)
                outputs_g_z = self.disc(gen_z)
                lossG = self.compute_gen_loss(outputs_g_z, gen_z)

                #real_loss_d = criterion(outputs_d_x, data)
                #fake_loss_d = criterion(outputs_d_z, gen_z.detach())
                    
                '''
                print "Fake LOSS:............."
                print outputs_d_z[0,0,:10,:10]
                print ".............."
                print gen_z[0,0,:10,:10]
                print "fake_loss_d:==>",fake_loss_d
                '''

                #lossG = criterion(gen_z, outputs_g_z)
                #loss = lossD + lossG
                lossG.backward()
                g_opti.step()
                balance = (opt.gamma * real_loss_d - fake_loss_d).data[0]
                opt.k += opt.lambda_k * balance
                #k = min(max(0, k), 1)
                opt.k = max(min(1, opt.k), 0)
                convg_measure = real_loss_d.data[0] + np.abs(balance)
                measure_history.append(convg_measure)

                if self.global_step % opt.print_step == 0:
                    print("Step: %d, Epochs: %d, Loss D: %.9f, real_loss: %.9f,"
                          " fake_loss: %.9f, Loss G: %.9f, k: %f, M: %.9f, lr:%.9f" % (self.global_step, i,
                                                                                       lossD.data[0], real_loss_d.data[0],
                                                                                       fake_loss_d.data[0], lossG.data[0],
                                                                                       opt.k, convg_measure, lr))
                    self.generate(gen_z, outputs_d_x, self.global_step)
                if opt.lr_update_type == 1:
                    lr = opt.lr * 0.95 ** (self.global_step // opt.lr_update_step)
                elif opt.lr_update_type == 2:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        lr *= 0.5
                elif opt.lr_update_type == 3:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        lr = min(lr * 0.5, opt.lr_lower_boundary)
                else:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        cur_measure = np.mean(measure_history)
                        if cur_measure > prev_measure * 0.9999:
                            lr = min(lr * 0.5, opt.lr_lower_boundary)
                        prev_measure = cur_measure

                for p in g_opti.param_groups + d_opti.param_groups:
                    p['lr'] = lr

                # g_opti = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=lr)
                # d_opti = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=lr)

                ''' 
                if self.global_step % lr_update_step == lr_update_step - 1:
                    cur_measure = np.mean(measure_history)
                    if cur_measure > prev_measure * 0.9999:
                        lr *= 0.5
                        g_opti = torch.optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=lr)
                        d_opti = torch.optim.Adam(disc.parameters(), betas=(0.5, 0.999), lr=lr)
                    prev_measure = cur_measure
                ''' 

                if self.global_step % 1000 == 0:
                    self.save_models(self.global_step)
            
                #convergence_history.append(convg_measure)
                self.global_step += 1


def generative_experiments(obj):
    z = []
    for inter in range(10):
        z0 = np.random.uniform(-1, 1, opt.h)
        z10 = np.random.uniform(-1, 1, opt.h)
        def slerp(val, low, high):
            omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
            so = np.sin(omega)
            if so == 0:
                return (1.0 - val) * low + val * high # L'Hopital's rule/LERP
            return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

        z.append(z0)
        for i in range(1, 9):
            z.append(slerp(i * 0.1, z0, z10))
        z.append(z10.reshape(1, opt.h))
    z = [_.reshape(1, opt.h) for _ in z]
    z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
    print(z_var.size())
    if opt.cuda:
        z_var = z_var.cuda()
    gen_z = obj.gen(z_var)
    obj.generate(gen_z, None, 'gen_1014_slerp_%d' % opt.load_step, 10)

    '''
    # Noise arithmetic 
    for i in range(5):
        sum_z = z[i] + z
        z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
        print z_var.size()
        if opt.cuda:
            z_var = z_var.cuda()
        gen_z = obj.gen(z_var)
        obj.generate(gen_z, None, 'gen_1014_slerp_%d'%i)
    '''
           

if __name__ == "__main__":
    obj = BEGAN()
    if opt.train:
        obj.train()
    elif opt.bp:
        print('starting bp')
        obj.birthday_paradox(10, opt.b_size)
    else:
        generative_experiments(obj)

