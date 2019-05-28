import utils
import networks_improve
import minibatch_discrimination

import torch 
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchsummary import summary


batchSize = 100

def smooth_labels(labels, offset):
	return labels + offset

def real_data_labels(size):
	data = Variable(torch.ones(size, 1))
	return data
	
def fake_data_labels(size):
	data = Variable(torch.zeros(size, 1))
	return data
	
		
def train_discriminator(optimizer, real_data, fake_data, discriminator, generator, loss):
		
	optimizer.zero_grad()
		
	# D(X^(i))
	# forward pass
	prediction_real = discriminator(real_data)
	
	# minimize loss function
	# manually, this would require differentiating loss function
	# and then updating weights by the appropriate amount,
	# but this is taken care of with .backward()
	# how much does real data differ from ones (real data class label)
	
	# One sided label smoothing (Salimans et. al, 2016)
	error_real = loss(prediction_real, smooth_labels(real_data_labels(real_data.size(0)),-0.1))
	error_real.backward()
		
	prediction_fake = discriminator(fake_data)
		
	error_fake = loss(prediction_fake, fake_data_labels(fake_data.size(0)))
	error_fake.backward()
		
	optimizer.step()
		
	return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
	
	optimizer.zero_grad()
	
	prediction = discriminator(fake_data)
	
	# we want data to be classified as real
	error = loss(prediction, Variable(0.9*torch.ones(real_data.size(0))))
	error.backward()
	
	optimizer.step()
	
	return error


if __name__ == "__main__":

	#TODO: data augmentation!
	#TODO: normalize with mean and stddev
	cat_data = utils.CatsDataset("./data/cats", transforms.Compose(
										[ transforms.Grayscale(num_output_channels=1),
										  transforms.Resize((32,32)),		
										  transforms.ToTensor(),
										  transforms.Normalize((.5,), (.5,))
										  
										  
										])
						  )
						  
	# Sample random batches from dataset  
	data_loader = torch.utils.data.DataLoader(cat_data, batch_size=batchSize, shuffle=True)
	
	
	# Get the number of sampled batches
	num_batches = len(data_loader)
	
	discriminator = networks_improve.DiscriminatorNet()
	generator = networks_improve.GeneratorNet()
	
	
	#TODO: change to sgd for D in order to follow original Goodfellow paper
	#TODO: experiment with learning rates
	d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002)
	g_optimizer = optim.Adam(generator.parameters(), lr=0.00002)
	
	# used in Goodfellow paper
	loss = nn.BCELoss()
	
	# TODO: test me out -> random search	
	# used in Goodfellow paper
	d_steps = 1
	
	num_epochs = 100
	
	num_test_samples = 16
	test_noise = networks_improve.noise(num_test_samples)


	logger = utils.Logger(model_name='GAN_improve', data_name='Cats')

	discriminator_loss = []
	generator_loss = []

	for epoch in range(num_epochs):
		for n_batch, real_batch in enumerate(data_loader):

			# 1. Train Discriminator
			
			#generate real data
			real_data = Variable(networks_improve.images_to_vectors(real_batch))
			
			# Generate fake data
			fake_data = generator(networks_improve.noise(real_data.size(0))).detach()
			
			# Train D
			d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
																	real_data, fake_data, discriminator, generator, loss)
			
			discriminator_loss.append(d_error)

			# =========================================

			# 2. Train Generator
			
			# Generate fake data
			fake_data = generator(networks_improve.noise(real_batch.size(0)))
			
			# Train G
			g_error = train_generator(g_optimizer, fake_data)
			
			generator_loss.append(g_error)
			
			# =========================================
			
			# Log error
			logger.log(d_error, g_error, epoch, n_batch, num_batches)

			# Display Progress
			if (n_batch) % 50 == 0 and (n_batch) > 50:
				utils.display.clear_output(True)
				
				#utils.plt.plot(discriminator_loss, label = 'D_error')
				#utils.plt.plot(generator_loss, label = 'G_error')
				#utils.plt.legend()
				#utils.plt.show()
				
				#fig = utils.plt.figure(figsize=(8,2))
				#cols = 8
				#rows = 2
				
				
				# Display Images
				test_images = networks_improve.vectors_to_images(generator(test_noise)).data.cpu()
				
				#for i in range(1, cols*rows):
				#	img = test_images[i]
				#	fig.add_subplot(rows, cols, i)
				#	utils.plt.imshow(img.permute(1,2,0).squeeze(), cmap='gray')
				#utils.plt.show()
					
				
				logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
				# Display status Logs
				logger.display_status(
					epoch, num_epochs, n_batch, num_batches,
					d_error, g_error, d_pred_real, d_pred_fake
				)
			# Model Checkpoints
			logger.save_models(generator, discriminator, epoch)

