import os, glob
import seaborn as sns
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

""" Used to create UMAPs from representations saved during training"""

dir = "results/bf" # Group folder, i.e. the folder that contains folders with runs

def analyze_mean_neighbour_dist(data, nn = 50, highest_n = 100):
	"""
	Calculates the mean dist value for the UMAP.
	@param data: array, the representation from the loaded pt file
	@param nn: int, the maximum amount of nearest neighbours
	@param highest_n: int, index for partitioning the mean_dist
	"""
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(data)
	distances, indices = nbrs.kneighbors(data)
	
	mean_dist = np.mean(distances[:,1:], axis = 1)
	ind = np.argpartition(mean_dist, -highest_n)[-highest_n:]
# 	print("mean",mean_dist)
# 	print("topn",mean_dist[ind])
	result_top = np.mean(mean_dist[ind])
	mean = np.mean(mean_dist)
	return result_top/mean
	
def create_umap(name):
	"""
	Creates a UMAP from .log and .pt files within 'name'
	@param name: string, name of the folder to work on (dir)
	"""
	global dir
	direc = dir + "/" + name + "/"
	os.chdir(direc + "representations/")
	
	# Palette size of 2x50 required. 1-49 for labeled nat data, 51-100 for labeled syn data, 50 for unlabeled nat data
	palette = sns.color_palette("Blues_d", 30)# Syn data in blue
	palette.extend(sns.dark_palette("purple", 20)) # Unimportant, just a filler
	palette.extend(sns.color_palette("Reds_d", 30))# Nat data in red
	palette.extend(sns.dark_palette("purple", 20))# Unimportant, just a filler
	palette[49]="#50B689"# Unlabeled nat data in green
	# print("size of palette " + str(len(palette)))
	
	for file in glob.glob("*.pt"):
		if "000" in file:
			representation = torch.load(file)
			tarfile = file[:-3] # Removes the .pt ending
			tarfile = "tar" + tarfile[4:] + ".log"
			all_targets = []
			with open(tarfile, "r") as f:
				for tar in f:
					all_targets.append(float(tar.strip()))

			sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
			reducer = umap.UMAP()
			print(name, file, "mean_dist: ",analyze_mean_neighbour_dist(representation.cpu()))
			
			embedding = reducer.fit_transform(representation.cpu())
			
			print("scattering")
			# print(all_targets)
			plt.scatter(embedding[:, 0], embedding[:, 1], c=[palette[int(y-1)] for y in all_targets], alpha=0.8)
			plt.gca().set_aspect('equal', 'datalim')
			plt.title('UMAP projection of cell data', fontsize=24);
			plt.savefig("./umap_" + str(file[4:-3]) + ".png")
			plt.clf()
	os.chdir("../../../../")

Parallel(n_jobs=4)(delayed(create_umap)(name) for name in os.listdir(dir))
	