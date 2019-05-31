from __future__ import print_function


import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
import os
import torch
import torch.utils.data
import torchvision.datasets as dset
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

from PIL import Image

#HACK
import torchvision.transforms.functional as F

class Logger:

	def __init__(self, model_name, data_name):
		self.model_name = model_name
		self.data_name = data_name

		self.comment = '{}_{}'.format(model_name, data_name)
		self.data_subdir = '{}/{}'.format(model_name, data_name)

		# TensorBoard
		self.writer = SummaryWriter(comment=self.comment)

	def log(self, d_error, g_error, epoch, n_batch, num_batches):

		# var_class = torch.autograd.variable.Variable
		if isinstance(d_error, torch.autograd.Variable):
			d_error = d_error.data.cpu().numpy()
		if isinstance(g_error, torch.autograd.Variable):
			g_error = g_error.data.cpu().numpy()

		step = Logger._step(epoch, n_batch, num_batches)
		self.writer.add_scalar(
			'{}/D_error'.format(self.comment), d_error, step)
		self.writer.add_scalar(
			'{}/G_error'.format(self.comment), g_error, step)

	def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
		'''
		input images are expected in format (NCHW)
		'''
		if type(images) == np.ndarray:
			images = torch.from_numpy(images)
		
		if format=='NHWC':
			images = images.transpose(1,3)
		

		step = Logger._step(epoch, n_batch, num_batches)
		img_name = '{}/images{}'.format(self.comment, '')

		# Make horizontal grid from image tensor
		horizontal_grid = vutils.make_grid(
			images, normalize=normalize, scale_each=True)
		# Make vertical grid from image tensor
		nrows = int(np.sqrt(num_images))
		grid = vutils.make_grid(
			images, nrow=nrows, normalize=True, scale_each=True)

		# Add horizontal images to tensorboard
		self.writer.add_image(img_name, horizontal_grid, step)

		# Save plots
		self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

	def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
		out_dir = './data/images/{}'.format(self.data_subdir)
		Logger._make_dir(out_dir)

		# Plot and save horizontal
		fig = plt.figure(figsize=(16, 16))
		plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
		plt.axis('off')
		if plot_horizontal:
			display.display(plt.gcf())
		self._save_images(fig, epoch, n_batch, 'hori')
		plt.close()

		# Save squared
		fig = plt.figure()
		plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
		plt.axis('off')
		self._save_images(fig, epoch, n_batch)
		plt.close()

	def _save_images(self, fig, epoch, n_batch, comment=''):
		out_dir = './data/images/{}'.format(self.data_subdir)
		Logger._make_dir(out_dir)
		fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
														 comment, epoch, n_batch))

	def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
		
		# var_class = torch.autograd.variable.Variable
		if isinstance(d_error, torch.autograd.Variable):
			d_error = d_error.data.cpu().numpy()
		if isinstance(g_error, torch.autograd.Variable):
			g_error = g_error.data.cpu().numpy()
		if isinstance(d_pred_real, torch.autograd.Variable):
			d_pred_real = d_pred_real.data
		if isinstance(d_pred_fake, torch.autograd.Variable):
			d_pred_fake = d_pred_fake.data
		
		
		print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
			epoch,num_epochs, n_batch, num_batches)
			 )
		print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
		print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

	def save_models(self, generator, discriminator, epoch):
		out_dir = './data/models/{}'.format(self.data_subdir)
		Logger._make_dir(out_dir)
		torch.save(generator.state_dict(),
				   '{}/G_epoch_{}'.format(out_dir, epoch))
		torch.save(discriminator.state_dict(),
				   '{}/D_epoch_{}'.format(out_dir, epoch))

		def load_models(self, gen_model, discr_model):
				device = torch.device("cuda")
				discr_dir = './data/models/models/VGAN/MNIST/D_epoch_18'
				gen_dir = './data/models/models/VGAN/MNIST/G_epoch_18'

				gen_model.load_state_dict(torch.load(gen_dir))
				gen_model.to(device)

				discr_model.load_state_dict(torch.load(discr_dir))
				discr_model.to(device)

				return model

	def close(self):
		self.writer.close()

	# Private Functionality

	@staticmethod
	def _step(epoch, n_batch, num_batches):
		return epoch * num_batches + n_batch

	@staticmethod
	def _make_dir(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

class CatsDataset(Dataset):
	""" Cat faces dataset. """
	
	def __init__(self, root_dir, transform = None):
		
		"""
			root_dir (string): Directory with images
			transform (callable, opt): Optional transformations
		"""
		
		self.root_dir = root_dir
		self.transform = transform
		lista = []
		for dirs, subdirs, fnames in os.walk(self.root_dir):
			for fname in fnames:
				lista.append(fname)
		self.img_names = lista
		
		
	def __len__(self):
		return len(self.img_names)
		
	def __getitem__(self,idx):
		img_name = os.path.join(self.root_dir, self.img_names[idx])
		
		img = Image.open(img_name)
		
		
		if self.transform:
			img = self.transform(img)
			
		return img
"""
class CarsDataset(Dataset):
	""" Cars dataset """
	
	def __init__(self, root_dir, transform=None):
		
		
		self.root_dir = root_dir
		self.transform = transform
		lista[]
		for dirs, subdirs, fnames in os.walk(self.root_dir):
			for fname in fnames:
				lista.append(fname)
		self.img_names = lista
		
	def __len__(self):
		return len(self.img_names)
		
	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.img_names[idx])
		
		img = Image.open(img_name)
		
		if self.transform:
			img = self.transform(img)
			
		return img
"""
