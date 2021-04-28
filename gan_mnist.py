import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from gan_definition import *

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0


BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for

lib.print_model_settings(locals().copy())


def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    samples = netG(noise)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
    )

def calc_gradient_penalty(netD, real_data, fake_data):
    #print(real_data.size())
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.ones([])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)


# ============== dataloaders definition begin ======================
# replace this section when you want to change a data set
training_data = datasets.FashionMNIST(
    'tmp/data',
    train=True,
    transform=ToTensor(),
    download=True
    )
testing_data = datasets.FashionMNIST(
    'tmp/data',
    train=False,
    transform=ToTensor(),
    download=True
)

train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
# testing data is for 
test_dataloader = DataLoader(testing_data, 4*BATCH_SIZE, shuffle=True)

# netG_gt = Generator()
# gt_path = ''
# gt_cp = torch.load(gt_path)
# netG_gt.load_state_dict(gt_cp['p_g'])

# train_dataloader = Dataset_from_GAN(netG_gt, BATCH_SIZE)
# test_dataloader = Dataset_from_GAN(netG_gt, 4*BATCH_SIZE)
# ============== dataloaders definition end ======================


train_iterator = iter(train_dataloader)
test_iterator = iter(test_dataloader)

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    netD.requires_grad_(True)
    for iter_d in range(CRITIC_ITERS):
        try:
            real_data = next(train_iterator)[0] # discard the label
        except StopIteration:
            #when data are used up, restart 
            train_iterator = iter(train_dataloader)
            real_data = next(train_iterator)[0]
        if use_cuda:
            real_data = real_data.cuda(gpu)
        # reshape to be consistent with G's output
        real_data = real_data.view(-1, OUTPUT_DIM)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        fake_data = netG(noise).detach() ### .detach() is fucking important
        D_fake = netD(fake_data)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)
        gradient_penalty.backward(retain_graph=False)

        D_cost = D_fake - D_real + gradient_penalty
        est_w1 = D_real - D_fake
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    netD.requires_grad_(False)

    netG.requires_grad_(True)
    netG.zero_grad()
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    fake_data = netG(noise)
    G = netD(fake_data)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()
    netG.requires_grad_(False)

    # Write logs and save samples
    lib.plot.plot('tmp/mnist/time', time.time() - start_time)
    lib.plot.plot('tmp/mnist/train disc cost', D_cost.detach().cpu().numpy())
    lib.plot.plot('tmp/mnist/train gen cost', G_cost.detach().cpu().numpy())
    lib.plot.plot('tmp/mnist/Estimated W1 Distance', est_w1.detach().cpu().numpy())

    # Calculate disc loss on true data and generate samples every 100 iters
    if iteration % 100 == 99:
        test_disc_costs = []
        try:
            imgs = next(test_iterator)[0]
        except StopIteration:
            test_iterator = iter(test_dataloader)
            imgs = next(test_iterator)[0]
        if use_cuda:
            imgs = imgs.cuda(gpu)

            D = netD(imgs)
            _test_disc_cost = -D.mean().cpu().numpy()
            test_disc_costs.append(_test_disc_cost)
        lib.plot.plot('tmp/mnist/test disc cost', np.mean(test_disc_costs))

        generate_image(iteration, netG)

    # Write logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()

    lib.plot.tick()

    # save checkpoint every 100 iterations
    if iteration%100 == 99:
        checkpoint = {
            'iteration': iteration,
            'p_g': netG.state_dict(),
            'p_d': netD.state_dict(),
            'o_g': optimizerG.state_dict(),
            'o_d': optimizerD.state_dict()
        }
        torch.save(checkpoint, 'tmp/mnist/checkpoints/cur.pth')
    
    # save checpoint without overwrite every 1000 iterations
    if iteration%1000 == 999:
        checkpoint = {
            'iteration': iteration,
            'p_g': netG.state_dict(),
            'p_d': netD.state_dict(),
            'o_g': optimizerG.state_dict(),
            'o_d': optimizerD.state_dict()
        }
        torch.save(checkpoint, 'tmp/mnist/checkpoints/iteration{}.pth'.format(iteration))
