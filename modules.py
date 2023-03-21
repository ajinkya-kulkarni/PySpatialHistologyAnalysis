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
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy import signal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from contextlib import redirect_stdout

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist import random_label_cmap
RandomColormap = random_label_cmap()

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

def perform_analysis(rgb_image, threshold_probability = 0.5, overlap_threshold = 0.3):
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

			# number_of_tiles = model._guess_n_tiles(rgb_image)

			labels, detailed_info = model.predict_instances(normalize(rgb_image), n_tiles = (10, 10, 1),prob_thresh = threshold_probability, nms_thresh = overlap_threshold, show_tile_progress = False)

	except:
		raise ValueError('Error predicting instances using StarDist2D model')

	return labels, detailed_info

##########################################################################

def compute_kde_heatmap(centroids, label_image, max_num_centroids=100):
	"""
	Computes a kernel density estimate (KDE) heatmap of the input centroids.

	Parameters:
	centroids (list or numpy array): A list or numpy array of centroid coordinates.
	label_image (numpy array): A labeled image where each blob has a unique integer label.
	max_num_centroids (int): The maximum number of centroids to use.

	Returns:
	A numpy array representing the KDE evaluated on a 2D grid.
	"""
	# Compute subsample factor
	num_centroids = len(centroids)
	if num_centroids <= max_num_centroids:
		subsample_factor = 1
	else:
		subsample_factor = int(np.ceil(num_centroids / max_num_centroids))
	
	# Subsample centroid coordinates
	centroids = centroids[::subsample_factor]

	# Create kernel density estimate of centroids
	kde = gaussian_kde(centroids.T)

	# Define grid for heatmap
	shape = label_image.shape
	x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	grid_coords = np.vstack([x.ravel(), y.ravel()])

	# Evaluate kernel density estimate on grid
	kde_heatmap = kde(grid_coords).reshape(shape)

	return kde_heatmap

##########################################################################

def plot_neighborhood_analysis(label_image):
	"""
	Plots a heatmap of the pairwise distance between the centroids of each label in the input label image.

	Parameters:
	label_image (numpy array): A labeled image where each blob has a unique integer label.

	Returns:
	A matplotlib figure object.
	"""
	# Compute centroids of each label
	centroids = []
	for label in np.unique(label_image)[1:]:
		coords = np.where(label_image == label)
		centroid = (np.mean(coords[0]), np.mean(coords[1]))
		centroids.append(centroid)

	# Compute pairwise distance matrix between centroids
	distances = squareform(pdist(centroids))

	# Create heatmap plot of distances
	fig, ax = plt.subplots()
	im = ax.imshow(distances, cmap='viridis')
	cbar = ax.figure.colorbar(im, ax=ax)
	cbar.ax.set_ylabel('Pairwise Distance', rotation=-90, va="bottom")
	ax.set_title('Pairwise Distance Heatmap of Label Centroids')

	return distances
	
##########################################################################

def cluster_labels_by_criterion(criterion_list, label_image, n_clusters = 3, n_init = 20):
	"""
	Clusters the labels in a label image based on a criterion.

	Parameters:
	criterion_list (numpy array): A 1D numpy array of criterion for each label in the image.
	label_image (numpy array): A labeled image where each blob has a unique integer label.
	n_clusters (int): The number of clusters to use for KMeans clustering.
	n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.

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

def convolve(image, kernel):
	"""
	Perform convolution on a binary image with a kernel of any size

	Parameters:
		image (np.ndarray): binary image to perform convolution on
		kernel (np.ndarray): kernel of any size

	Returns:
		np.ndarray: binary image after convolution
	"""

	# Convert the input image to a valid data type
	image = np.array(image, dtype=np.float32)

	# Get the shape of the image
	i_h, i_w = image.shape

	# Get the shape of the kernel
	k_h, k_w = kernel.shape

	# Check if the kernel is of odd size
	if k_h % 2 == 0 or k_w % 2 == 0:
		raise ValueError("Kernel must be of odd size")

	# Check if the kernel size is smaller than the image dimensions
	if k_h > i_h or k_w > i_w:
		raise ValueError("Kernel size must be smaller than image dimensions")

	# Pad the image with the pixels along the edges
	pad_h = int((k_h - 1) / 2)
	pad_w = int((k_w - 1) / 2)
	image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

	# Get the total number of elements in the kernel
	total_elements = k_h * k_w

	# Perform convolution
	result = signal.convolve2d(image, kernel / total_elements, mode='same', boundary='fill', fillvalue=0)

	return result[pad_h:-pad_h, pad_w:-pad_w]

##########################################################################

def make_plots(rgb_image, labels, detailed_info, modified_labels_rgb_image, Local_Density, criterion, cluster_labels, cluster_number, SIZE = "3%", PAD = 0.08, title_PAD = 10, DPI = 300, ALPHA = 1):

	# Create the figure and axis objects
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), dpi = DPI)

	# Display uploaded image
	im = axs[0, 0].imshow(rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs[0, 0])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[0, 0].set_title('Uploaded Image', pad = title_PAD)
	# Turn off axis ticks and labels
	axs[0, 0].set_xticks([])
	axs[0, 0].set_yticks([])
	cax.remove()

	######################
	
	# Display labelled image
	im = axs[0, 1].imshow(labels, cmap = RandomColormap)
	# Add a colorbar
	divider = make_axes_locatable(axs[0, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[0, 1].set_title('Segmented Nuclei, N=' + str(len(detailed_info['points'])), pad = title_PAD)
	# Turn off axis ticks and labels
	axs[0, 1].set_xticks([])
	axs[0, 1].set_yticks([])
	cax.remove()

	######################

	# # Display the clustered blob labels figure
	# # Create a mask where cluster_labels is equal to 0
	# mask = np.where(cluster_labels == 0, np.nan, 1)
	# # Apply the mask to cluster_labels
	# cluster_labels = cluster_labels * mask
	# # Plot the image with NaN regions
	im_clusters = axs[1, 0].imshow(cluster_labels, alpha=ALPHA, cmap='viridis')
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 0])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_clusters, cax=cax)

	if criterion == 'eccentricity':
		criterion_name = 'Roundness'
	if criterion == 'area':
		criterion_name = 'Size'

	axs[1, 0].set_title(str(cluster_number - 1) + ' nuclei groups by ' + criterion_name, pad = title_PAD)
	# Turn off axis ticks and labels
	axs[1, 0].set_xticks([])
	axs[1, 0].set_yticks([])
	# get tick locations and update to integers
	cb.set_ticks(range(cluster_number))
	tick_locs = cb.get_ticks()
	int_tick_labels = [int(tick) for tick in tick_locs]
	cb.set_ticklabels(int_tick_labels)
	# cax.remove()

	######################

	# Display the density map figure
	im_density = axs[1, 1].imshow(Local_Density, vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2, cmap='Spectral_r')
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_density, cax=cax)
	axs[1, 1].set_title('Local Nuclei Density', pad = title_PAD)
	# Turn off axis ticks and labels
	axs[1, 1].set_xticks([])
	axs[1, 1].set_yticks([])

	######################

	# # Remove the last subplot in the bottom row
	# fig.delaxes(axs[2, 1])
	
	return fig

##########################################################################