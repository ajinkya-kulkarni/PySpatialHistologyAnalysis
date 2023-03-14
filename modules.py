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

def compute_kde_heatmap(centroids, label_image, subsample_factor=100):
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

def cluster_labels_by_eccentricity(eccentricities, label_image, n_clusters=5):
	"""
	Clusters the labels in a label image based on their eccentricity values.

	Parameters:
	label_image (numpy array): A labeled image where each blob has a unique integer label.
	eccentricities (numpy array): A 1D numpy array of eccentricity values for each label in the image.
	n_clusters (int): The number of clusters to use for KMeans clustering.

	Returns:
	A numpy array representing the cluster labels evaluated on the input label image.
	"""
	# Reshape the eccentricities array
	eccentricities = eccentricities.ravel()

	# Perform KMeans clustering on the eccentricity values
	kmeans = KMeans(n_clusters=n_clusters).fit(eccentricities.reshape(-1, 1))

	# Assign cluster labels to each label in the input image
	cluster_labels = np.zeros_like(label_image)
	for i, label in enumerate(np.unique(label_image)[1:]):
		cluster_labels[label_image == label] = kmeans.labels_[i]

	return cluster_labels

##########################################################################

def ripley_k(centroids, r_max, num_points=100):
	"""
	Computes Ripley's K function for a set of points in 2D space.

	Parameters:
	centroids (list or numpy array): A list or numpy array of centroid coordinates.
	r_max (float): The maximum radius to compute K function for.
	num_points (int): The number of points to use for the grid in each dimension.

	Returns:
	A tuple containing the distance bins and the values of the K function.
	"""
	# Compute pairwise distance between centroids
	distances = cdist(centroids, centroids)

	# Define range of radii to compute K function for
	r = np.linspace(0, r_max, num_points)

	# Compute K function values for each radius
	K_values = []
	for radius in r:
		N_pairs = (distances <= radius).sum() - len(centroids)
		K = N_pairs / len(centroids)
		K_values.append(K)

	# Compute expected K function values for a homogeneous Poisson point process
	lambda_ = len(centroids) / ((centroids.max(axis=0) - centroids.min(axis=0)).prod())
	expected_K_values = np.pi * r**2 * lambda_

	# Compute cumulative K function values
	K_cumulative = np.cumsum(K_values) / len(centroids)
	expected_K_cumulative = np.cumsum(expected_K_values) / len(centroids)

	# Compute corrected K function values
	K_corrected = np.sqrt(K_cumulative / np.pi) - r
	expected_K_corrected = np.sqrt(expected_K_cumulative / np.pi) - r

	return r, K_corrected - expected_K_corrected

##########################################################################