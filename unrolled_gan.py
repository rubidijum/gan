import utils
import unrolled_networks

import torch
import copy
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchsummary import summary

batch_size = 100

def real_data_labels(size):
	data = Variable(torch.ones(size,1))
	return data
	
def fake_data_labels(size):
	data = Variable(torch.zeros(size,1))
	return data
	
def smooth_labels(labels, offset):
	return labels + offset
	
def train_discriminator(optimizer, real_data, fake_data, discriminator, generator, loss, unrolled=False):
	
	optimizer.zero_grad()
	
	# Train on real data:
	prediction_real = discriminator(real_data)
	
	error_real = loss(prediction_real, smooth_labels(real_data_labels(real_data.size(0)), -0.1))

	
	# Train of fake data:
	prediction_fake = discriminator(fake_data)
	
	error_fake = loss(prediction_fake, fake_data_labels(fake_data.size(0)))
	
	# construct computational graph if training is done in unrolled step
	# create_graph -> enables to backprop w.r.t gradients
	error_D = error_real + error_fake
	error_D.backward(create_graph=unrolled)
	
	# update parameters 
	optimizer.step()
	
	return error_D, prediction_real, prediction_fake
	
def train_generator(fake_data, G_optimizer, loss=None, D_optimizer=None, unrolled_steps=0):
	
	G_optimizer.zero_grad()
	
	if(unrolled_steps > 0):
		if(loss == None or D_optimizer == None):
			return
		# save discriminator copy - we don't want to train this D, 
		# it's only used for G's training -> we will restore it later
		unrolled_D = copy.deepcopy(D)
		for i in range(unrolled_steps):
			train_discriminator(D_optimizer, real_data, fake_data, unrolled_D, G, loss, unrolled=True)
			# train 'unrolled' discriminator for unrolled_steps 
		
	# predict with 'unrolled' D
	prediction = unrolled_D(fake_data)
	
	error_G = loss(prediction, real_data_labels(fake_data.size(0)))
	error_G.backward()
	
	G_optimizer.step()
	
	return error_G
	
if __name__ == "__main__":
	
	cat_data = utils.CatsDataset("./data/cats", transforms.Compose(
													[
													 transforms.ToTensor(),
													 transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
													]
												))
												
	data_loader = torch.utils.data.DataLoader(cat_data, batch_size=batch_size, shuffle=True)
	
	batch_size = len(data_loader)
	
	D = unrolled_networks.DiscriminatorNet()
	G = unrolled_networks.GeneratorNet()
	
	D_optimizer = optim.Adam(D.parameters(), lr=0.0002)
	G_optimizer = optim.Adam(G.parameters(), lr=0.0002)
	
	loss = nn.BCELoss()
	
	unrolled_steps = [5,10]
	
	num_epochs = 100
	
	num_test_samples = 16
	test_noise = unrolled_networks.noise(num_test_samples)
	
	logger = utils.Logger(model_name="Unrolled_GAN", data_name="Cats")
	
	D_loss = []
	G_loss = []
	
	for epoch in range(num_epochs):
		for n_batch, real_batch in enumerate(data_loader):
			
			# Train D
			
			real_data = Variable(unrolled_networks.images_to_vectors(real_batch))
			
			fake_data = G(unrolled_networks.noise(real_data.size(0))).detach()
			
			D_error, D_prediction_real, D_prediction_fake = train_discriminator(D_optimizer, real_data, fake_data, D, G, loss)
	
			D_loss.append(D_error)
			
			# =========================================
			
			# Train G - search no. of unroll steps
			
			fake_data = G(unrolled_networks.noise(real_data.size(0)))
			
			G_error = train_generator(fake_data, G_optimizer, loss, D_optimizer, unrolled_steps=5)
			
			G_loss.append(G_error)
			
			# =========================================
			
			logger.log(D_error, G_error, epoch, n_batch, num_batches)
			
			if (n_batch) % 50 == 0 and n_batch > 50:
				
				utils.display.clear_output(True)
				
				test_images = unrolled_networks.vectors_to_images(G(test_noise)).data.cpu()
				
				logger.log_images(test_images, num_test_samples, epoch, n_batch)
				
				logger.display_status(epoch, num_epochs, n_batch, num_batches, D_error, G_error, D_prediction_real, D_prediction_fake)
			
			logger.save_models(G, D, epoch)
	
	
	
	
	
	
	
