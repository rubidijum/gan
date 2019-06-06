from __future__ import print_function

#%matplotlib inline
import utils
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
from IPython.display import HTML

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
num_ch = 3

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
                                #transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5), (.5,.5,.5))
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

logger = utils.Logger(model_name="DCGAN", data_name="Cats")
save_path = os.mkdir("./images/")

print("Starting training...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        # Update D
        # maximize log(D(x)) + log(1-D(G(z)))

        # train with all-real batch
        netD.zero_grad()

        # format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # forward pass through the Discriminator
        output = netD(real_cpu).view(-1)

        # calculate loss on all-real batch 
        errD_real = criterion(output, label)
        # calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # generator batch of vectors
        noise = torch.randn(b_size, z_size, 1, 1, device=device)

        # generate all-fake batch with Generator
        fake = netG(noise)
        # this batch is fake for the Discriminator 
        label.fill_(fake_label)

        # forward pass all-fake batch through the Discriminator
        output = netD(fake.detach()).view(-1)

        # calculate Discriminator's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # calculate gradients for D on the all-fake batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # add the gradients from the all-real and the all-fake batch
        errD = errD_real + errD_fake

        # update D with the Adam optimizer
        optimizerD.step()


        # Update Generator's network 
        # maximize log(D(G(z))
        netG.zero_grad()
        # fake label is real for the Generator
        label.fill_(real_label)

        # perform another forward pass on the updated Discriminator
        output = netD(fake).view(-1)

        # calculate Generator's loss based on the Discriminator's output
        errG = criterion(output, label)
        # and then calculate gradients for the Generator
        errG.backward()
        D_G_z2 = output.mean().item()

        # update Generator with the Adam optimizer
        optimizerG.step()
        
        logger.log(errD, errG, epoch, i, len(dataloader))

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            logger.log_images(fake, 16, epoch, i, batch_size);

        iters += 1
"""
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
"""
