import torch 
from utils import Logger
import utils
import networks
import networks_improve
import os

if __name__ == "__main__":
	
	
	l = utils.Logger("demonstrate", "demonstrate")
	
	
	
	architectures = ['VGAN', 'GAN_improve']
	
	data = ['Cats', 'Cars']
	
	print("""Choose architecture: 
				[1] Vanilla GAN
				[2] Improved GAN (one-sided label smoothing, minibatch discrimination layer)
				[3] Unrolled GAN (not yet implemented)
				[4] DCGAN (not yet implemented)
			""")
	
	arch = architectures[int(input())-1]
			
	print("""Choose data:
				[1] Cats
				[2] Cars (not yet implemented)
			""")
	
	dataname = data[int(input())-1]
	
	if (arch == "VGAN"):
		G = networks.GeneratorNet()
	elif (arch == "GAN_improve"):
		G = networks_improve.GeneratorNet()
	
	for _,_,fnames in os.walk("./data/models/{}".format(arch)):
		model_list = fnames #sorted(fnames, reverse = True)
	model_list = filter(lambda x: "G" in x, model_list)
	max_G = (sorted(model_list, key = lambda x: int(x[x.rindex("_")+1:]))[-1])
	max_epoch = max_G[max_G.rindex("_")+1:]
	
	print("Choose epoch: [0 - " + str(max_epoch) + "]")
	
	epoch_num = input()
	
	l.load_G(arch, dataname, int(epoch_num), G)
	
	
	rows = 5
	cols = 5
	
	num_samples = 25
	
	print("Enter number of iterations: ")
	num_iters = int(input())
	
	for itr in range(num_iters):
		
		print("Iteration number " + str(itr))
		
		if(arch == "GAN_improve"):
			test_noise = networks_improve.noise(num_samples)
			fake_data = G(test_noise).detach()
			fake_imgs = networks_improve.vectors_to_images(fake_data)
		elif(arch == "VGAN"):
			test_noise = networks.noise(num_samples)
			fake_data = G(test_noise).detach()
			fake_imgs = networks.vectors_to_images(fake_data)
		else:
			print("ERROR!");
		
		fig = utils.plt.figure(figsize=(5,5))
		
		for i in range(1, rows*cols+1):
			print(i)
			img = fake_imgs[i-1]
			fig.add_subplot(rows, cols, i, xticks=[], yticks=[])
			print(img.size())
			if(arch == "GAN_IMPROVE"):
				utils.plt.imshow(img.permute(1,2,0))
			elif(arch == "VGAN"):
				utils.plt.imshow(img.permute(1,2,0).squeeze(), cmap='gray')
		utils.plt.axis('off')
		utils.plt.show()
		
		
