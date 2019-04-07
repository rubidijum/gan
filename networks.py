import utils

import torch 
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

class DiscriminatorNet(torch.nn.Module):
	
	"""
		Architecture description 
	"""
	
	def __init__(self):
		super().__init__()
		
		n_features = 4096
		n_out = 1
		
		self.hidden0 = nn.Sequential(
			nn.Linear(n_features, 2048),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden1 = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		
		self.hidden3 = nn.Sequential(
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		
		
		# TODO: change activation to maxout as in
		# original paper (Goodfellow)
		self.out = nn.Sequential(
			torch.nn.Linear(256, n_out),
			torch.nn.Sigmoid()
		)
	
	def forward(self, x):
		x = self.hidden0(x) 
		x = self.hidden1(x) 
		x = self.hidden2(x) 
		x = self.hidden3(x)  
		x = self.out(x) 
		return x
		
def images_to_vectors(images):
	return images.view(images.size(0), 4096)
	
def vectors_to_images(vectors):
	return vectors.view(vectors.size(0), 1, 64, 64)
	
class GeneratorNet(torch.nn.Module):
	
	
	def __init__(self):
		super().__init__()
		n_features = 100
		n_out = 4096
		
		self.hidden0 = nn.Sequential(
			nn.Linear(n_features, 256),
			nn.LeakyReLU(0.2)
		)
		
		self.hidden1 = nn.Sequential(
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2)
		)
		
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2)
		)
		
		self.hidden3 = nn.Sequential(
			nn.Linear(1024, 2048),
			nn.LeakyReLU(0.2)
		)
		
		# TODO: change activation
		self.out = nn.Sequential(
			nn.Linear(2048, n_out),
			nn.Tanh()
		)
		
	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.hidden3(x)
		x = self.out(x)
		return x
		
# change this a bit
def noise(size):
	n = Variable(torch.randn(size,100))
	return n
	

