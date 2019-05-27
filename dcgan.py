from __future__ import print_function

#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

seed = 42

random.seed(seed)
torch.manual_seed(seed)

# the path to the root of the dataset folder
dataroot = "./data/"

# the number of worker thread for loading the data with the DataLoader
workers = 2

# batch size used in training. DCGAN paper uses a batch of 128
batch_size = 128

image_size = 64

# number of channels in the training images
num_ch = 1

# size of generator input
z_size = 100

# size of feature maps in generator
gen_feat_map_size = 64

# size of feature maps in discriminator
discr_feat_map_size = 64

num_epochs = 5

# learning rate for optimizers 
learning_rate = 0.0002

# beta1 hyperparam for ADAM optimizer
beta1 = 0.5

ngpu = 1

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((.5,), (.5,))
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

real_batch = next(iter(dataloader))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
                nn.ConvTranspose2d(z_size, 
                                   gen_feat_map_size*8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(gen_feat_map_size*8),
                nn.ReLU(True),

                nn.ConvTranspose2d(gen_feat_map_size*8,
                                   gen_feat_map_size*4,
                                   4, 2, 1, bias = False),
                nn.BatchNorm2d(gen_feat_map_size*4),
                nn.ReLU(True),

                nn.ConvTranspose2d(gen_feat_map_size*4,
                                   gen_feat_map_size*2,
                                   4, 2, 1, bias = False),
                nn.BatchNorm2d(gen_feat_map_size*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(gen_feat_map_size*2,
                                   gen_feat_map_size,
                                   4, 2, 1, bias=False),
                nn.BatchNorm2d(gen_feat_map_size),
                nn.ReLU(True),

                nn.ConvTranspose2d(gen_feat_map_size, num_ch,
                                   4, 2, 1, bias=False),
                nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)

class Discriminator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                nn.Conv2d(num_ch, discr_feat_map_size, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(discr_feat_map_size, discr_feat_map_size*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(discr_feat_map_size * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(discr_feat_map_size * 2, discr_feat_map_size * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(discr_feat_map_size * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(discr_feat_map_size * 4, discr_feat_map_size * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(discr_feat_map_size * 8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(discr_feat_map_size * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.main(x)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, z_size, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting training...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_cpu).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()

        D_x = output.mean().item()

        noise = torch.randn(b_size, z_size, 1, 1, device=device)

        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()

        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()


        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Plotting
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualisation of generator's progression
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Real vs. fake
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
