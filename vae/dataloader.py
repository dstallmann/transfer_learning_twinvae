import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops
import pandas as pd
import numpy as np
import random
import subprocess

from learner import batch_size, img_size, seed, img_noise, crop_size, datatype_case, dynamic_data, ratio

# Unsure if all of this has global scope (and can be removed here) or class-wide scope and needs to stay
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# random.seed(seed)

all_images = []

class CustomDataset(Dataset):
	"""
	Class for a customized data set to improve read times and accessibility of the data and to store the preprocessed data.
	"""
	def __init__(self, csv_path):
		"""
		Initializes the data set and applies all data augmentations that can safely be done before the actual training.
		@param csv_path: path to the csv file that contains the image data informations
		"""
		# Transforms
		self.to_tensor = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			# transforms.RandomAffine(0, translate=(2 / img_size, 2 / img_size), scale=None, shear=None, resample=False, fillcolor=0),
			# transforms.RandomRotation(3),
			
			transforms.RandomResizedCrop(img_size, scale=(crop_size, crop_size), ratio=(1.0, 1.0)),
			transforms.ToTensor(), ])
		# transforms.Normalize((0.5,), (0.5,))])
		self.data_info = pd.read_csv(csv_path, header=None)
		# First column contains the image paths
		self.image_arr = np.asarray(self.data_info.iloc[:, 0])
		# Second column is the labels
		self.label_arr = np.asarray(self.data_info.iloc[:, 1])

		self.data_len = len(self.data_info.index)

		loaded_images = []
		for image_path in self.image_arr:
			img_as_img = Image.open(prefix + image_path[2:])
			img_as_img.load()
			img_as_img = img_as_img.convert('L')  # Luminosity
			loaded_images.append(img_as_img)
		self._loaded_images = loaded_images

	def __getitem__(self, index):
		"""
		Function to get an image from the custom data set. Applies post processing augmentations to the data.
		@param index: int, the index of the image-label combination
		@return: returns the image as a tensor variable together with the label for that image, being 0 for unlabeled cases.
		"""
		# image_name = self.image_arr[index] # Get image name from csv
		image_label = self.label_arr[index]  # Get image label from csv
		img_as_img = self._loaded_images[index]

		angle = random.choice([0, 90])  # Don't need -90, because that is achieved in combination with the flips.
		img_as_img = TF.rotate(img_as_img, angle)
		
		#print(img_as_img.mode)
		img_as_tensor = self.to_tensor(img_as_img)  # Already scaled to [0,1]
		
		noise_map = torch.autograd.Variable(torch.randn(img_size, img_size) * img_noise)
		img_as_tensor = img_as_tensor.add(noise_map)

		return img_as_tensor, image_label

	def __len__(self):
		return self.data_len

prefix = os.getcwd().replace("\\", "/")[:-4]  # gets the current path up to /vae and removes the /vae to get to the data directory
print(prefix)

if dynamic_data == True:
	data_generation_process = subprocess.Popen(["python", "../data_generator/image_generator.py", "--data_case", datatype_case, "--ratio", str(ratio)], shell=False)
	data_generation_process.wait()

num_workers = 0  # Use just 0 or 1 workers to prevent heavy memory overhead and slower loading
if datatype_case == "bf":
 	syn_train_set = CustomDataset(prefix + '/data/128p_bf' + ('_dyn' if dynamic_data else '') + '/train_gen.csv')
if datatype_case == "pc":
	syn_train_set = CustomDataset(prefix + '/data/128p_pc' + ('_dyn' if dynamic_data else '') + '/train_gen.csv')
syn_train_load = torch.utils.data.DataLoader(dataset=syn_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

if datatype_case == "bf":
	syn_test_set = CustomDataset(prefix + '/data/128p_bf' + ('_dyn' if dynamic_data else '') + '/test_gen.csv')
if datatype_case == "pc":
	syn_test_set = CustomDataset(prefix + '/data/128p_pc' + ('_dyn' if dynamic_data else '') + '/test_gen.csv')
syn_test_load = torch.utils.data.DataLoader(dataset=syn_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

if datatype_case == "bf":
	nat_train_set = CustomDataset(prefix + '/data/128p_bf/train.csv')
if datatype_case == "pc":
	nat_train_set = CustomDataset(prefix + '/data/128p_pc/train.csv')
nat_train_load = torch.utils.data.DataLoader(dataset=nat_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

if datatype_case == "bf":
	nat_test_set = CustomDataset(prefix + '/data/128p_bf/test.csv')
if datatype_case == "pc":
	nat_test_set = CustomDataset(prefix + '/data/128p_pc/test.csv')
nat_test_load = torch.utils.data.DataLoader(dataset=nat_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
