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

import streamlit as st

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

@st.cache_resource
def perform_analysis(rgb_image, threshold_probability):
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

			labels, detailed_info = model.predict_instances(normalize(rgb_image), n_tiles = (10, 10, 1),prob_thresh = threshold_probability, nms_thresh = 0.3, show_tile_progress = False)

	except:
		raise ValueError('Error predicting instances using StarDist2D model')

	return labels, detailed_info

##########################################################################

def cluster_labels_by_criterion(criterion_list, label_image, n_clusters, n_init=20):
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

	# Get unique labels excluding the background label (0)
	unique_labels = np.unique(label_image)[1:]

	# Extract the criteria for the foreground labels
	foreground_criteria = criterion_list[unique_labels - 1]

	# Perform KMeans clustering on the foreground criteria
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(foreground_criteria.reshape(-1, 1))

	# Assign cluster labels to each label in the input image
	cluster_labels = np.zeros_like(label_image)
	for i, label in enumerate(unique_labels):
		cluster_labels[label_image == label] = kmeans.labels_[i] + 1

	return cluster_labels

##########################################################################

def make_plots(rgb_image, labels, detailed_info, Local_Density, area_cluster_labels, area_cluster_number, eccentricity_cluster_labels, eccentricity_cluster_number, SIZE = "3%", PAD = 0.2, title_PAD = 15, DPI = 300, ALPHA = 1):

	fig, axs = plt.subplot_mosaic([['b', 'c'], ['d', 'e']], figsize=(18, 15), layout="constrained", dpi = DPI, gridspec_kw={'hspace': 0, 'wspace': 0.2})
	
	# Display labelled image
	im = axs['b'].imshow(labels, cmap = RandomColormap)
	# Add a colorbar
	divider = make_axes_locatable(axs['b'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs['b'].set_title('Segmented Nuclei, N=' + str(len(detailed_info['points'])), pad = title_PAD)
	# Turn off axis ticks and labels
	axs['b'].set_xticks([])
	axs['b'].set_yticks([])
	cax.remove()

	######################

	# Display the density map figure
	im_density = axs['c'].imshow(Local_Density, vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2, cmap='viridis')
	# Add a colorbar
	divider = make_axes_locatable(axs['c'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_density, cax=cax)
	axs['c'].set_title('Local Nuclei Density', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['c'].set_xticks([])
	axs['c'].set_yticks([])
	# Calculate the tick locations for Low, Medium, and High
	low_tick = 0
	med_tick = 0.5
	high_tick = 1
	# Set ticks and labels for Low, Medium, and High
	cb.set_ticks([low_tick, med_tick, high_tick])
	cb.set_ticklabels(['Low', 'Medium', 'High'])

	######################

	# # Display the area clustered blob labels figure

	area_cluster_labels = np.where(area_cluster_labels==0, np.nan, area_cluster_labels)

	im_area_cluster_labels = axs['d'].imshow(area_cluster_labels, alpha=ALPHA, cmap = 'Set1')
	# Add a colorbar
	divider = make_axes_locatable(axs['d'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_area_cluster_labels, cax=cax)
	axs['d'].set_title(str(area_cluster_number) + ' nuclei groups by area', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['d'].set_xticks([])
	axs['d'].set_yticks([])
	
	# Calculate the tick locations for Low, Medium, and High
	low_tick = 1
	high_tick = area_cluster_number
	# Set ticks and labels for Low, Medium, and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])
	# cax.remove()

	######################

	# # Display the eccentricity clustered blob labels figure

	eccentricity_cluster_labels = np.where(eccentricity_cluster_labels==0, np.nan, eccentricity_cluster_labels)

	im_eccentricity_cluster_labels = axs['e'].imshow(eccentricity_cluster_labels, alpha=ALPHA, cmap = 'Set1')
	# Add a colorbar
	divider = make_axes_locatable(axs['e'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_eccentricity_cluster_labels, cax=cax)
	axs['e'].set_title(str(eccentricity_cluster_number) + ' nuclei groups by roundness', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['e'].set_xticks([])
	axs['e'].set_yticks([])

	# Calculate the tick locations for Low, Medium, and High
	low_tick = 1
	high_tick = eccentricity_cluster_number
	# Set ticks and labels for Low, Medium, and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])
	# cax.remove()

	######################

	# Remove the last subplot in the bottom row
	# fig.delaxes(axs['f'])
	
	return fig

##########################################################################

def run_test():

	pass
