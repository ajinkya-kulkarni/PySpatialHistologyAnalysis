#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

##########################################################################

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from contextlib import redirect_stdout

from stardist.models import StarDist2D
from csbdeep.utils import normalize

##########################################################################

def read_image(filename):
	"""
	Reads an image from a file and returns it as a NumPy array.

	Parameters:
	filename (str): The path to the image file.

	Returns:
	numpy.ndarray: The image as a NumPy array.
	"""
	try:
		img = Image.open(filename)
		rgb_image = img.convert('RGB')
		rgb_image = np.array(rgb_image)
	except:
		raise ValueError('Error reading image file')
	
	return rgb_image

##########################################################################

def perform_analysis(rgb_image):
	"""
	Performs object detection on an RGB image using the StarDist2D model.

	Parameters:
	rgb_image (numpy.ndarray): An RGB image as a NumPy array.

	Returns:
	numpy.ndarray: The labeled image as a NumPy array.
	"""
	try:
		with redirect_stdout(open(os.devnull, "w")) as f:

			model = StarDist2D.from_pretrained('2D_versatile_he')

			labels, detailed_info = model.predict_instances(normalize(rgb_image))

	except:
		raise ValueError('Error predicting instances using StarDist2D model')

	return labels, detailed_info

##########################################################################

def compute_kde_heatmap(centroids, label_image, subsample_factor):
	"""
	Computes a kernel density estimate (KDE) heatmap of the input centroids.

	Parameters:
	centroids (list or numpy array): A list or numpy array of centroid coordinates.
	label_image (numpy array): A labeled image where each blob has a unique integer label.
	subsample_factor (int): The factor by which to subsample the centroid coordinates.

	Returns:
	A numpy array representing the KDE evaluated on a 2D grid.
	"""
	# Subsample centroid coordinates
	centroids = centroids[::subsample_factor]

	# Create kernel density estimate of centroids
	kde = gaussian_kde(centroids.T)

	# Define grid for heatmap
	shape = label_image.shape
	x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	grid_coords = np.vstack([x.ravel(), y.ravel()])

	# Evaluate kernel density estimate on grid
	z = kde(grid_coords).reshape(shape)

	return z

##########################################################################

def cluster_labels_by_criterion(criterion_list, label_image, n_clusters = 3, n_init = 20):
	"""
	Clusters the labels in a label image based on a criterion.

	Parameters:
	label_image (numpy array): A labeled image where each blob has a unique integer label.
	criterion (numpy array): A 1D numpy array of criterion for each label in the image.
	n_clusters (int): The number of clusters to use for KMeans clustering.

	Returns:
	A numpy array representing the cluster labels evaluated on the input label image.
	"""
	# Reshape the criterion_list array
	criterion_list = criterion_list.ravel()

	# Perform KMeans clustering on the criterion values

	kmeans = KMeans(n_clusters = n_clusters, n_init = n_init).fit(criterion_list.reshape(-1, 1))

	# Assign cluster labels to each label in the input image
	cluster_labels = np.zeros_like(label_image)
	for i, label in enumerate(np.unique(label_image)[1:]):
		cluster_labels[label_image == label] = kmeans.labels_[i]

	return cluster_labels

##########################################################################

def make_plots(rgb_image, modified_labels_rgb_image, modified_labels, kde_heatmap, criterion, cluster_labels, cluster_number, SIZE = "3%", PAD = 0.07, DPI = 300):

	# Create the figure and axis objects
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi = DPI)

	# Display uploaded image
	im = axs[0, 0].imshow(rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs[0, 0])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[0, 0].set_title('Uploaded Image')
	# Turn off axis ticks and labels
	axs[0, 0].set_xticks([])
	axs[0, 0].set_yticks([])
	cax.remove()
	
	# Display labelled image
	im = axs[0, 1].imshow(modified_labels_rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs[0, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[0, 1].set_title('Segmented Nuclei')
	# Turn off axis ticks and labels
	axs[0, 1].set_xticks([])
	axs[0, 1].set_yticks([])
	cax.remove()

	# Overlay the labels image and KDE heatmap
	im = axs[1, 0].imshow(modified_labels, cmap = 'binary', zorder = 1)
	im_heatmap = axs[1, 0].imshow(kde_heatmap / kde_heatmap.max(), cmap='coolwarm', vmin = 0, vmax = 1, alpha=0.8, zorder = 2)
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 0])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_heatmap, cax=cax)
	axs[1, 0].set_title('Nuclei Density')
	# Turn off axis ticks and labels
	axs[1, 0].set_xticks([])
	axs[1, 0].set_yticks([])

	# Display the clustered blob labels
	im = axs[1, 1].imshow(cluster_labels, cmap='viridis')
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[1, 1].set_title('Nuclei grouped into ' + str(cluster_number - 1) + ' classes by ' + criterion)
	# Turn off axis ticks and labels
	axs[1, 1].set_xticks([])
	axs[1, 1].set_yticks([])
	cax.remove()
	
	return fig

##########################################################################