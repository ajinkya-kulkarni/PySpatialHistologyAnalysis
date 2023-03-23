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

@st.cache_data
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

def colorize_labels(labels):
	"""
	Takes a grayscale label image and colorizes each label with a random RGB color,
	except for the background label which is left as black. Returns the colorized RGB
	image.

	Parameters
	----------
	labels : numpy.ndarray
		A grayscale label image where each pixel has an integer value corresponding to
		its label. The background label should have a value of 0.

	Returns
	-------
	numpy.ndarray
		An RGB image where each label in the input image is colored with a random RGB
		color, except for the background label which is left as black.
	"""

	# Generate a random RGB color for each label
	num_labels = len(np.unique(labels))
	colors = np.random.randint(0, 256, (num_labels, 3))

	# Create a blank RGB image
	h, w = labels.shape
	modified_labels_rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

	# Color each label with its random color
	for label in np.unique(labels):
		if label == 0:
			# Background label - leave as black
			continue
		else:
			# Random color for non-background labels
			color = tuple(colors[label - 1])
			modified_labels_rgb_image[labels == label] = color
	
	return modified_labels_rgb_image

##########################################################################

from skimage.util import view_as_windows

def count_nuclei_per_window(labelled_image, window_number = 200):
	"""
	Counts the number of nuclei per sliding window of the specified number in a labelled image.

	Parameters:
	labelled_image (ndarray): A 2D numpy array containing the labelled image, where each nucleus has a unique non-zero label.
	window_number (int): The number of windows in each direction. Defaults to 20.

	Returns:
	ndarray: A 2D numpy array containing the number of nuclei per window, normalized by the window size squared.
	"""
	rows, cols = labelled_image.shape

	# Calculate the size of each window
	window_size = int(np.ceil(cols / window_number))

	# Pad the image with zeros to ensure that the window shape is a multiple of the image shape
	pad_rows = window_size * window_number - rows
	pad_cols = window_size * window_number - cols
	labelled_image_padded = np.pad(labelled_image, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

	# Create a view of the padded image as a stack of windows
	window_shape = (window_size, window_size)
	windows = view_as_windows(labelled_image_padded, window_shape)

	# Count the number of nuclei per window using nested loops
	nuclei_per_window = np.zeros((window_number, window_number))
	for i in range(nuclei_per_window.shape[0]):
		for j in range(nuclei_per_window.shape[1]):
			window = windows[i, j]
			nuclei = np.unique(window)
			nuclei = np.delete(nuclei, np.where(nuclei == 0)) # Remove background label
			nuclei_per_window[i, j] = len(nuclei) / window_size**2

	# Upscale the result to the size of the input image
	nuclei_per_image = np.zeros_like(labelled_image, dtype=np.float64)
	for i in range(window_number):
		for j in range(window_number):
			x0 = i * window_size
			x1 = x0 + window_size
			y0 = j * window_size
			y1 = y0 + window_size
			nuclei_per_image[x0:x1, y0:y1] = nuclei_per_window[i, j]

	return nuclei_per_image

##########################################################################

def calculate_local_density(labels):
	"""
	Calculates local density of an input array.

	Parameters:
		labels (numpy array): 2D array of labels

	Returns:
		Local_Density (numpy array): 2D array of local density values

	"""
	# Calculate window size as 10% of the minimum dimension of the input array
	window_size = int(0.1 * min(labels.shape[0], labels.shape[1]))

	# Apply a blur filter to the input array using the calculated window size
	# This effectively averages the pixel values in a local neighborhood around each pixel
	Local_Density = cv2.blur(labels, (window_size, window_size), cv2.BORDER_DEFAULT)

	# Normalize the Local_Density array by dividing it by its maximum value
	# This ensures that the values are between 0 and 1
	# Use np.full and np.nan to fill in any NaN values in the result of division where the maximum value is 0
	Local_Density = np.divide(Local_Density, Local_Density.max(), out=np.full(Local_Density.shape, np.nan), where=Local_Density.max() != 0)
	
	return Local_Density

##########################################################################

def bin_property_values(labels, property_values, n_bins):
	"""
	Bin the property values into n_bins equally spaced bins and assign the binned values to each label.

	Args:
	- labels (ndarray): An array of integers representing the labels for each property value.
	- property_values (ndarray): An array of floats representing the property values.
	- n_bins (int): The number of equally spaced bins to create.

	Returns:
	- binned_values (ndarray): An array of floats representing the binned values for each label.
	"""
	# Compute the histogram of the property values
	hist, bins = np.histogram(property_values, bins=n_bins)
	
	# Create an array to store the binned values
	binned_values = np.zeros_like(labels, dtype=float)
	binned_values.fill(np.nan)
	
	# Assign the binned values to each label
	for i, p in enumerate(property_values):
		binned_values[labels == i+1] = np.digitize(p, bins[:-1])
	
	return binned_values

##########################################################################

def make_plots(modified_labels_rgb_image, detailed_info, Local_Density, area_cluster_labels, area_cluster_number, roundness_cluster_labels, roundness_cluster_number, SIZE = "3%", PAD = 0.2, title_PAD = 15, DPI = 300, ALPHA = 1):

	fig, axs = plt.subplot_mosaic([['b', 'c'], ['d', 'e']], figsize=(18, 15), layout="constrained", dpi = DPI, gridspec_kw={'hspace': 0, 'wspace': 0.2})

	## Display RGB labelled image

	im = axs['b'].imshow(modified_labels_rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs['b'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs['b'].set_title(str(len(detailed_info['points'])) + ' segmented Nuclei', pad = title_PAD)
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
	high_tick = 1
	# Set ticks and labels for Low, Medium, and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])

	######################

	# # Display the area clustered blob labels figure

	im_area_cluster_labels = axs['d'].imshow(area_cluster_labels, alpha=ALPHA, cmap = 'brg')
	# Add a colorbar
	divider = make_axes_locatable(axs['d'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_area_cluster_labels, cax=cax)
	axs['d'].set_title(str(area_cluster_number) + ' nuclei groups by Area', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['d'].set_xticks([])
	axs['d'].set_yticks([])
	
	# Calculate the tick locations for Low, and High
	low_tick = 1
	high_tick = area_cluster_number
	# Set ticks and labels for Low and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])
	# cax.remove()

	######################

	# # Display the roundness clustered blob labels figure

	im_roundness_cluster_labels = axs['e'].imshow(roundness_cluster_labels, alpha=ALPHA, cmap = 'brg')
	# Add a colorbar
	divider = make_axes_locatable(axs['e'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_roundness_cluster_labels, cax=cax)
	axs['e'].set_title(str(roundness_cluster_number) + ' nuclei groups by Roundness', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['e'].set_xticks([])
	axs['e'].set_yticks([])

	# Calculate the tick locations for Low and High
	low_tick = 1
	high_tick = roundness_cluster_number
	# Set ticks and labels for Low and High
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
