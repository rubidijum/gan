from __future__ import print_function

import os
import torch
import torch.utils.data
import torchvision.datasets as dset
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

#dataroot = "/home/aleksandar/gan/data/"
#batchsize = 

class CatsDataset(Dataset):
	""" Cat faces dataset. """
	
	def __init__(self, root_dir, transform = None):
		
		"""
			root_dir (string): Directory with images
			transform (callable, opt): Optional transformations
		"""
		
		self.root_dir = root_dir
		self.transform = transform
		self.img_names = [name for name in os.listdir(self.root_dir)]
		
	def __len__(self):
		return len(self.img_names)
		
	def __getitem__(self,idx):
		img_name = os.path.join(self.root_dir, self.img_names[idx])
		img = io.imread(img_name)
		
		if self.transform:
			img = self.transform(img)
			
		return img
		

								
if __name__ == "__main__":
	
	cats_dataset = CatsDataset("/home/aleksandar/gan/data/cats", 
	transforms.Compose([
								transforms.ToTensor(),
								#transforms.Normalize((.5, .5, .5), (.5, .5, .5))
	])
	)
	
	fig = plt.figure()

	#dataloader = DataLoader(cats_dataset, batch_size=5,
							#shuffle=True, num_workers=4)
							
	#for i, sample_batched in enumerate(dataloader):
	#	plt.figure()
	#	if i == 3:
	#		grid = utils.make_grid(sample_batched)
	#		plt.imshow(grid)
	#		plt.show()
	#		break

	for i in range(len(cats_dataset)):
		
		sample = cats_dataset[i]
		
		print(sample.shape)
		
		ax = plt.subplot(1, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i))
		ax.axis('off')
		
		
		
		plt.imshow(sample.permute(1,2,0))
		#plt.imshow(sample)
		
		if i == 3:
			plt.show()
			break
	
	dataLoad = torch.utils.data.DataLoader(cats_dataset, batch_size=10, shuffle=True)
	
	for i, image in enumerate(dataLoad):
		
		
		if i == 3:
			for img in image:
				print("shp" + str(img.shape))
				img = img.permute(1,2,0)
				
				plt.imshow(img)
				
				plt.show()
				break
