import utils
import networks

import torch 
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchsummary import summary


batchSize = 200

def real_data_labels(size):
	data = Variable(torch.ones(size, 1))
	return data
	
def fake_data_labels(size):
		data = Variable(torch.zeros(size, 1))
		return data
		
def train_discriminator(optimizer, real_data, fake_data, discriminator, generator, loss):
		
		optimizer.zero_grad()
		
		prediction_real = discriminator(real_data)
		
		error_real = loss(prediction_real, real_data_labels(real_data.size(0)))
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
	error = loss(prediction, real_data_labels(prediction.size(0)))
	error.backward()
	
	return error


if __name__ == "__main__":

	#TODO: data augmentation!
	#TODO: normalize with mean and stddev
	cat_data = utils.CatsDataset("./data/cats", transforms.Compose(
										[ transforms.Grayscale(num_output_channels=1),
										  transforms.ToTensor(),
										  #transforms.Normalize((.5), (.5))										  
										])
						  )
						  
	# Sample random batches from dataset  
	data_loader = torch.utils.data.DataLoader(cat_data, batch_size=batchSize, shuffle=True)
	
	
	# Get the number of sampled batches
	num_batches = len(data_loader)
	
	discriminator = networks.DiscriminatorNet()
	generator = networks.GeneratorNet()
	
	#print(discriminator)
	#print(generator)
	
	summary(discriminator, (1,4096))
	summary(generator, (1,100))
	
	d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002)
	g_optimizer = optim.Adam(generator.parameters(), lr=0.00002)
	
	loss = nn.BCELoss()
	
	# TODO: test me out -> random search	
	d_steps = 1
	num_epochs = 2
	
	num_test_samples = 16
	test_noise = networks.noise(num_test_samples)


logger = utils.Logger(model_name='VGAN', data_name='MNIST')

discriminator_loss = []
generator_loss = []

for epoch in range(num_epochs):
	for n_batch, (real_batch) in enumerate(data_loader):

		# 1. Train Discriminator
		
		real_data = Variable(networks.images_to_vectors(real_batch))
		#print(real_data[0].shape)
		#real_data_show = real_data[0].reshape(3,64,64)
		#utils.plt.imshow(real_data_show.permute(1,2,0))
		#utils.plt.show()
		if torch.cuda.is_available(): real_data = real_data.cuda()
		# Generate fake data
		fake_data = generator(networks.noise(real_data.size(0))).detach()
		#print(fake_data.shape)
		# Train D
		d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
																real_data, fake_data, discriminator, generator, loss)
		
		discriminator_loss.append(d_error)

		# 2. Train Generator
		# Generate fake data
		fake_data = generator(networks.noise(real_batch.size(0)))
		# Train G
		g_error = train_generator(g_optimizer, fake_data)
		# Log error
		logger.log(d_error, g_error, epoch, n_batch, num_batches)

		generator_loss.append(g_error)

		# Display Progress
		if (n_batch) % 100 == 0:
			utils.display.clear_output(True)
			
			utils.plt.plot(discriminator_loss, label = 'D_error')
			utils.plt.plot(discriminator_loss, label = 'G_error')
			utils.plt.legend()
			utils.plt.show()
			
			# Display Images
			test_images = networks.vectors_to_images(generator(test_noise)).data.cpu()
			logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
			# Display status Logs
			logger.display_status(
				epoch, num_epochs, n_batch, num_batches,
				d_error, g_error, d_pred_real, d_pred_fake
			)
		# Model Checkpoints
		logger.save_models(generator, discriminator, epoch)
