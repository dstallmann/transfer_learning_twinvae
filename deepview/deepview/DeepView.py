from deepview.embeddings import init_umap, init_inv_umap#Mapper, InvMapper
from deepview.fisher_metric import calculate_fisher

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings

class DeepView:

	def __init__(self, pred_fn, classes, max_samples, batch_size, data_shape, 
				 n=5, lam=0.65, resolution=100, cmap='tab10', interactive=True, 
				 title='DeepView', data_viz=None, mapper=None, inv_mapper=None, **kwargs):
		'''
		This class can be used to embed high dimensional data in
		2D. With an inverse mapping from 2D back into the sample
		space, the 2D space is sampled on a regular grid and a
		classification outcome for each point is visualized.
		
		Parameters
		----------
		pred_fn	: callable, function
			Function that takes a single argument, which is data to be classified
			and returns the prediction logits of the model.
			For an example, see the demo jupyter notebook:
			https://github.com/LucaHermes/DeepView/blob/master/DeepView%20Demo.ipynb
		classes	: list, tuple of str
			All classes that the classifier uses as a list/tuple of strings.
		max_samples	: int
			The maximum number of data samples that this class keeps for visualization.
			If the number of data samples passed via 'add_samples' exceeds this limit, 
			the oldest samples are deleted first.
		batch_size : int
			Batch size to use when calling the classifier
		data_shape : tuple, list (int)
			Shape of the input data.
		n : int
			Number of interpolations for distance calculation of two images.
		lam : float
			Weights the euclidian metric against the discriminative, classification-
			based, distance: eucl * lambda + discr * (1 - lambda).
			To put more emphasis on structural propertiues of the datapoints, use a higher lambda,
			a lower lambda will put emphasis on the classification properties of the datapoints.
		resolution : int
			Resolution of the classification boundary plot.
		cmap : str
			Name of the colormap to use for visualization.
			The number of distinguishable colors should correspond to n_classes.
			See here for the names: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
		interactive : bool
			If interactive is true, the show() method won't be blocking, so code execution will continue.
			Otherwise it will be blocking.
		title : str
			The title for the plot
		data_viz : callable, function
			A function that takes in a single sample in the original sample shape and visualizes it.
			This function will be called when the DeepView plot is clicked on, with the according data sample
			or synthesised sample at the click location.
		mapper : object
			An object that maps samples from the data input domain to 2D space. The object
			must have the methods of deepview.embeddings.Mapper. fit is called with a distance matrix
			of all data samples and transform is called with an image that should be projected to 2D. 
			Defaults to None, in this case UMAP is used.
		inv_mapper : object
			An object that maps samples from the 2D space to the data input domain. The object
			must have the methods of deepview.embeddings.Mapper. fit is called with 2D points and
			their according data samples. transform is called with 2D points that should be projected to data space. 
			Defaults to None, in this case deepview.embeddings.InvMapper is used.
		kwargs : dict
			Configuration for the embeddings in case they are not specifically given in mapper and inv_mapper.
			Defaults to deepview.config.py. 
			See UMAP-parameters here: 
			https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1167
		'''
		self.model = pred_fn
		self.classes = classes
		self.n_classes = len(classes)
		self.max_samples = max_samples
		self.batch_size = batch_size
		self.data_shape = data_shape
		self.n = n
		self.lam = lam
		self.resolution = resolution
		self.cmap = plt.get_cmap(cmap)
		self.discr_distances = np.array([])
		self.eucl_distances = np.array([])
		self.samples = np.empty([0, *data_shape])
		self.embedded = np.empty([0, 2])
		self.y_true = np.array([])
		self.y_pred = np.array([])
		self.classifier_view = np.array([])
		self.interactive = interactive
		self.title = title
		self.data_viz = data_viz
		self._init_mappers(mapper, inv_mapper, kwargs)

	@property
	def num_samples(self):
		return len(self.samples)

	@property
	def distances(self):
		'''
		Combines euclidian with discriminative fisher distances.
		Here the two distance measures are weighted with lambda
		to emphasise structural properties (lambda > 0.5) or
		to emphasise prediction properties (lambda < 0.5).
		'''
		eucl_scale = 1. / self.eucl_distances.max()
		fisher_scale = 1. / self.discr_distances.max()
		eucl = self.eucl_distances * eucl_scale * self.lam
		fisher = self.discr_distances * fisher_scale * (1.-self.lam)
		stacked = np.dstack((fisher, eucl))
		return stacked.sum(-1)

	def reset(self):
		'''
		Resets the state of DeepView to the point of initialization.
		'''
		self.discr_distances = np.array([])
		self.eucl_distances = np.array([])
		self.samples = np.empty([0, *self.data_shape])
		self.embedded = np.empty([0, 2])
		self.y_true = np.array([])
		self.y_pred = np.array([])
		self.classifier_view = np.array([])

	def close(self):
		'''
		Closes the matplotlib window, terminates DeepView.
		'''
		plt.close()

	def set_lambda(self, lam):
		'''
		Dynamically sets a new lambda and recomputes the
		embeddings, as the distances will also change.
		'''
		if self.lam == lam:
			return
		self.lam = lam
		self.update_mappings()

	def _init_mappers(self, mapper, inv_mapper, kwargs):
		if mapper is not None:
			self.mapper = mapper
		else:
			self.mapper = init_umap(kwargs)
		if inv_mapper is not None:
			self.inverse = inv_mapper
		else:
			self.inverse = init_inv_umap(kwargs)


	def _get_plot_measures(self):
		'''
		Computes the axis limits of the main plot by using
		min/max values of the 2D sample embedding and adding 10%
		on either side.
		'''
		ebd_min = np.min(self.embedded, axis=0)
		ebd_max = np.max(self.embedded, axis=0)
		ebd_extent = ebd_max - ebd_min

		# get extent of embedding
		x_min, y_min = ebd_min - 0.1 * ebd_extent
		x_max, y_max = ebd_max + 0.1 * ebd_extent
		return x_min, y_min, x_max, y_max

	def _init_plots(self):
		'''
		Initialises matplotlib artists and plots.
		'''
		if self.interactive:
			plt.ion()
		self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
		self.ax.set_title(self.title)
		self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
		self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]), 
			interpolation='gaussian', zorder=0, vmin=0, vmax=1)

		self.sample_plots = []

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', label=self.classes[c], 
				color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
			self.sample_plots.append(plot[0])

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', markeredgecolor=color, 
				fillstyle='none', ms=12, mew=2.5, zorder=1)
			self.sample_plots.append(plot[0])

		self.fig.canvas.mpl_connect('pick_event', self.show_sample)
		self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
		self.disable_synth = False
		self.ax.legend()
		#plt.show(block=False)

	def update_matrix(self, old_matrix, new_values, n_new, n_keep):
		'''
		When new distance values are calculated this merges old 
		and new distances into the same matrix.
		'''
		to_triu = np.triu(old_matrix, k=1)
		new_mat = np.zeros([self.num_samples, self.num_samples])
		new_mat[n_new:,n_new:] = to_triu[:n_keep,:n_keep]
		new_mat[:n_new] = new_values
		# update the old distance matrix
		return new_mat + new_mat.transpose()

	def update_mappings(self):
		print('Embedding samples ...')
		self.mapper.fit(self.distances)
		self.embedded = self.mapper.transform(self.distances)
		self.inverse.fit(self.embedded, self.samples)
		self.classifier_view = self.compute_grid()

	def add_samples(self, samples, labels):
		'''
		Adds samples points to the visualization.

		Parameters
		----------
		samples : array-like
			List of new sample points [n_samples, *data_shape]
		labels : array-like
			List of labels for the sample points [n_samples, 1]
		'''
		# prevent new samples to be bigger then maximum
		samples = samples[:self.max_samples]
		n_new = len(samples)

		# add new samples and remove depricated samples
		# and get predictions for the new samples
		self.samples = np.concatenate((samples, self.samples))[:self.max_samples]
		Y_preds = self.model(samples).argmax(axis=1)
		self.y_pred = np.concatenate((Y_preds, self.y_pred))[:self.max_samples]
		self.y_true = np.concatenate((labels, self.y_true))[:self.max_samples]

		# calculate new distances
		xs = samples
		ys = self.samples
		new_discr, new_eucl = calculate_fisher(self.model, xs, ys, self.n, 
			self.batch_size, self.n_classes)

		# add new distances
		n_keep = self.max_samples - n_new
		self.discr_distances = self.update_matrix(self.discr_distances, new_discr, n_new, n_keep)
		self.eucl_distances = self.update_matrix(self.eucl_distances, new_eucl, n_new, n_keep)

		# update mappings
		self.update_mappings()

	def compute_grid(self):
		'''
		Computes the visualisation of the decision boundaries.
		'''
		print('Computing decision regions ...')
		# get extent of embedding
		x_min, y_min, x_max, y_max = self._get_plot_measures()
		# create grid
		xs = np.linspace(x_min, x_max, self.resolution)
		ys = np.linspace(y_min, y_max, self.resolution)
		self.grid = np.array(np.meshgrid(xs, ys))
		grid = np.swapaxes(self.grid.reshape(self.grid.shape[0],-1),0,1)
		
		# map gridmpoint to images
		grid_samples = self.inverse(grid)
		n_points = self.resolution**2

		mesh_preds = np.zeros([n_points, self.n_classes])

		for i in range(0, n_points, self.batch_size):
			n_preds = min(i+self.batch_size, n_points)
			batch = grid_samples[i:n_preds]
			# add epsilon for stability
			mesh_preds[i:n_preds] = self.model(batch) + 1e-8

		self.mesh_classes = mesh_preds.argmax(axis=1)
		mesh_max_class = max(self.mesh_classes)

		# get color of gridpoints
		color = self.cmap(self.mesh_classes/mesh_max_class)
		# scale colors by certainty
		h = -(mesh_preds*np.log(mesh_preds)).sum(axis=1)/np.log(self.n_classes)
		h = (h/h.max()).reshape(-1, 1)
		# adjust brightness
		h = np.clip(h*1.2, 0, 1)
		color = color[:,0:3]
		color = (1-h)*(0.5*color) + h*np.ones(color.shape, dtype=np.uint8)
		decision_view = color.reshape(self.resolution, self.resolution, 3)
		return decision_view

	def get_mesh_prediction_at(self, x, y):
		x_idx = np.abs(self.grid[0,0] - x).argmin(0)
		y_idx = np.abs(self.grid[1,:,0] - y).argmin(0)
		mesh = self.mesh_classes.reshape([self.resolution]*2)
		return mesh[y_idx, x_idx]

	def show_sample(self, event):
		'''
		Invoked when the user clicks on the plot. Determines the
		embedded or synthesised sample at the click location and 
		passes it to the data_viz method, together with the prediction, 
		if present a groun truth label and the 2D click location.
		'''

		# when there is an artist attribute, a 
		# concrete sample was clicked, otherwise
		# show the according synthesised image
		if hasattr(event, 'artist'):
			artist = event.artist
			ind = event.ind
			xs, ys = artist.get_data()
			point = [xs[ind][0], ys[ind][0]]
			sample, p, t = self.get_artist_sample(point)
			title = '%s <-> %s' if p != t else '%s --- %s'
			title = title % (self.classes[p], self.classes[t])
			self.disable_synth = True
		elif not self.disable_synth:
			# workaraound: inverse embedding needs more points
			# otherwise it doens't work --> [point]*5
			point = np.array([[ event.xdata , event.ydata ]]*5)

			# if the outside of the plot was clicked, points are None
			if None in point[0]:
				return

			sample = self.inverse(point)[0]
			sample += abs(sample.min())
			sample /= sample.max()
			title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
			p, t = self.get_mesh_prediction_at(*point[0]), None
		else:
			self.disable_synth = False
			return

	
		# allowed shapes for images are (h,w), (h,w,3), (h,w,4)
		is_grayscale = len(sample.shape) == 2
		is_colored = len(sample.shape) == 3 and \
			(sample.shape[-1] == 3 or sample.shape[-1] == 4)

		if self.data_viz is not None:
			self.data_viz(sample, point, p, t)
		# try to show the image, if the shape allows it
		elif is_grayscale or is_colored:
			f, a = plt.subplots(1, 1)
			a.imshow(sample)
			a.set_title(title)
		else:
			warnings.warn("Data visualization not possible, as the data points have"
				"no image shape. Pass a function in the data_viz argument,"
				"to enable custom data visualization.")
			
	def get_artist_sample(self, point):
		'''
		Maps the location of an embedded point to it's image.
		'''
		sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
		sample = self.samples[sample_id]
		sample = sample + np.abs(sample.min())
		sample = sample / sample.max()
		yp, yt = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
		return sample, yp, yt

	def show(self):
		'''
		Shows the current plot.
		'''
		if not hasattr(self, 'fig'):
			self._init_plots()

		x_min, y_min, x_max, y_max = self._get_plot_measures()

		self.cls_plot.set_data(self.classifier_view)
		self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
		self.ax.set_xlim((x_min, x_max))
		self.ax.set_ylim((y_min, y_max))

		params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
		desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
		self.desc.set_text(desc)

		for c in range(self.n_classes):
			data = self.embedded[self.y_true==c]
			self.sample_plots[c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', c=self.cmap(c/9), 
			#	label=self.classes[c])

		for c in range(self.n_classes):
			data = self.embedded[np.logical_and(self.y_pred==c, self.y_true!=c)]
			self.sample_plots[self.n_classes+c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', markeredgecolor=self.cmap(c/9), 
			#	fillstyle='none', ms=200, linewidth=3)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		self.fig.show()
		#self.fig.canvas.manager.window.raise_()
		plt.show()

	@staticmethod
	def create_simple_wrapper(classify):
		'''
		Creates a basic wrapper function to be passed
		on DeepView initialization. Works with sklearn
		predict_proba methods.

		Arguments
		---------
		classify : function
			The function of a classifier called to
			predict class probabilities. Has to return
			a vector [batch, class probabilities]

		Returns
		-------
		wrapper : function
			Wrapper function that casts inputs to numpy
			array of dtype float32.
		'''
		def wrapper(x):
			x = np.array(x, dtype=np.float32)
			pred = classify(x)
			return pred
		return wrapper
