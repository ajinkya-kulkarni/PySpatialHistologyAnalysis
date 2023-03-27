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

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##########################################################################

from PIL import Image

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

from contextlib import redirect_stdout
from stardist.models import StarDist2D
from csbdeep.utils import normalize

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

from skimage.measure import regionprops
from sklearn.neighbors import KernelDensity

def weighted_kde_density_map(nucleus_mask, bandwidth='auto', kernel='gaussian', num_points = 500):
	"""
	Compute the weighted kernel density estimate (KDE) of the centroids of regions in a binary image.
	
	Parameters:
		nucleus_mask (ndarray): Binary image of nuclei, with 1's indicating nuclei and 0's indicating background.
		bandwidth (float or str, optional): The bandwidth of the KDE. If 'auto', use the rule of thumb bandwidth. Default is 'auto'.
		kernel (str, optional): The kernel function to use. Default is 'gaussian'.
		num_points (int, optional): The number of points to use in the density map. Default is 500.
		
	Returns:
		density_map (ndarray): The density map of the centroids of the nuclei.
	"""
	# Extract centroid locations and areas of each nucleus
	regions = regionprops(nucleus_mask)
	nucleus_centroids = np.array([region.centroid for region in regions])
	nucleus_areas = np.array([region.area for region in regions])

	# Compute the weighted KDE
	if bandwidth == 'auto':
		# Use the rule of thumb bandwidth selection
		num_nuclei = nucleus_centroids.shape[0]
		num_dimensions = nucleus_centroids.shape[1]
		bandwidth = num_nuclei**(-1.0/(num_dimensions+4)) * np.std(nucleus_centroids, axis=0).mean()

		# make bandwidth more refined than the calculated amount.
		bandwidth = float(0.5 * bandwidth)
		
	else:
		bandwidth = float(bandwidth)  # Ensure bandwidth is a float
	
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(nucleus_centroids, sample_weight=nucleus_areas)

	# Compute the size of the grid for the density map based on the desired number of points
	num_steps_x = int(np.ceil(np.sqrt(num_points * nucleus_mask.shape[1] / nucleus_mask.shape[0])))
	num_steps_y = int(np.ceil(np.sqrt(num_points * nucleus_mask.shape[0] / nucleus_mask.shape[1])))
	x_step_size = int(np.ceil(nucleus_mask.shape[1] / num_steps_x))
	y_step_size = int(np.ceil(nucleus_mask.shape[0] / num_steps_y))

	# Create a density map
	x_steps = int(np.ceil(nucleus_mask.shape[1] / x_step_size))
	y_steps = int(np.ceil(nucleus_mask.shape[0] / y_step_size))
	x_coords = np.arange(0, x_steps * x_step_size, x_step_size)
	y_coords = np.arange(0, y_steps * y_step_size, y_step_size)
	xx, yy = np.meshgrid(x_coords, y_coords)

	# Evaluate the KDE at each point in the grid to create the density map
	grid_points = np.vstack([yy.ravel(), xx.ravel()]).T
	density = np.exp(kde.score_samples(grid_points))
	density_map = density.reshape((y_steps, x_steps))

	return density_map

##########################################################################

from skimage.filters import rank

def mean_filter(labels):
	"""
	Calculates local density of an input array.
	# Compute label areas
	label_areas = np.bincount(label_image.flat)

	Parameters:
		labels (numpy array): 2D array of labels
	# Create kernel density estimate of centroids, weighted by label areas
	weights = label_areas[label_image] / np.sum(label_areas)
	kde = gaussian_kde(centroids.T, weights=weights)

	Returns:
		Local_Density (numpy array): 2D array of local density values
	# Define grid for heatmap
	shape = label_image.shape
	x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	grid_coords = np.vstack([x.ravel(), y.ravel()])

	"""
	# # Calculate window size as 10% of the minimum dimension of the input array
	# window_size = int(0.1 * min(labels.shape[0], labels.shape[1]))

	# # Apply a blur filter to the input array using the calculated window size
	# # This effectively averages the pixel values in a local neighborhood around each pixel
	# Local_Density = cv2.blur(labels, (window_size, window_size), cv2.BORDER_DEFAULT)

	labels = labels.astype('uint8')

	window_size = int(0.1 * min(labels.shape[0], labels.shape[1]))

	# Apply a mean filter to the input array using the calculated window size
	# This effectively averages the pixel values in a local neighborhood around each pixel
	Local_Density = rank.mean(labels, footprint = np.ones((window_size, window_size)))
	
	return Local_Density
	
##########################################################################

def normalize_density_maps(Local_Density):

	Normalized_Local_Density = np.divide(Local_Density, Local_Density.max(), out=np.full(Local_Density.shape, np.nan), where=Local_Density.max() != 0)

	return Normalized_Local_Density

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

from skimage import measure
from scipy.spatial import distance_matrix
import networkx as nx

def make_network_connectivity_graph(labelled_image, distance_threshold):
	"""
	This function creates a graph representation of a labelled image, where the nodes of the graph correspond to the nuclei in the image.
	Edges are added between nodes based on a distance threshold, and the weight of each edge is equal to the connectivity density between the corresponding nuclei. The nodes are then clustered based on their connectivity density, and each cluster is assigned a unique label.

	Inputs:

	labelled_image: a 2D image where each pixel has a unique integer label indicating the nucleus it belongs to.
	distance_threshold: a threshold value for the maximum distance between two nuclei centroids for them to be considered connected.
	Outputs:

	graph: a NetworkX graph object representing the nuclei in the image and their connectivity.
	node_labels: a 1D numpy array containing the cluster labels assigned to each nucleus in the image.
	"""

	# Extract properties of the nuclei
	properties = measure.regionprops_table(labelled_image, properties=['label', 'centroid'])
	nuclei_centroids = np.column_stack([properties['centroid-1'], properties['centroid-0']])
	num_nuclei = len(nuclei_centroids)

	# Calculate the distance matrix between the nuclei centroids
	distances = distance_matrix(nuclei_centroids, nuclei_centroids)

	# Create an empty graph
	graph = nx.Graph()

	# Add nodes to the graph
	for idx, centroid in enumerate(nuclei_centroids):
		graph.add_node(idx, pos=centroid)

	# Add edges to the graph based on connectivity density
	for i in range(num_nuclei):
		for j in range(i + 1, num_nuclei):
			if distances[i, j] <= distance_threshold:
				# Compute connectivity density between the nuclei
				connectivity_density = 2 / (distances[i, j] ** 2)
				# Add edge with weight equal to connectivity density
				graph.add_edge(i, j, weight=connectivity_density)

	# Cluster the nodes based on their connectivity density
	node_labels = np.zeros(num_nuclei, dtype=int)
	label_count = 0
	for node in graph.nodes():
		# Get the neighbors of the node
		neighbors = list(graph.neighbors(node))
		# Sort the neighbors by their connectivity density
		neighbors_sorted = sorted(neighbors, key=lambda n: graph.edges[(node, n)]['weight'], reverse=True)
		# Assign the same label to the node and its top-connected neighbors
		if node_labels[node] == 0:
			label_count += 1
			node_labels[node] = label_count
			for neighbor in neighbors_sorted:
				if node_labels[neighbor] == 0:
					node_labels[neighbor] = label_count

	# Add the node labels as a node attribute to the graph
	for node in graph.nodes():
		graph.nodes[node]['label'] = node_labels[node]

	return graph, node_labels

##########################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_plots(rgb_image, ModelSensitivity, modified_labels_rgb_image, detailed_info, Local_Density_mean_filter, Local_Density_KDE, area_cluster_labels, area_cluster_number, roundness_cluster_labels, roundness_cluster_number, SIZE = "3%", PAD = 0.2, title_PAD = 15, DPI = 300, ALPHA = 1):

	fig, axs = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'f']], figsize=(20, 30), layout="constrained", dpi = DPI, gridspec_kw={'hspace': 0, 'wspace': 0.2})

	## Display RGB labelled image

	im = axs['a'].imshow(rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs['a'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs['a'].set_title('H&E Image', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['a'].set_xticks([])
	axs['a'].set_yticks([])
	cax.remove()

	######################

	## Display RGB labelled image

	im = axs['b'].imshow(modified_labels_rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs['b'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs['b'].set_title(str(len(detailed_info['points'])) + ' segmented Nuclei, with σ=' + str(ModelSensitivity), pad = title_PAD)
	# Turn off axis ticks and labels
	axs['b'].set_xticks([])
	axs['b'].set_yticks([])
	cax.remove()

	######################

	# Display the density map figure

	im_density = axs['c'].imshow(Local_Density_mean_filter, vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2, cmap='cividis')
	# Add a colorbar
	divider = make_axes_locatable(axs['c'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_density, cax=cax)
	axs['c'].set_title('Nuclei density', pad = title_PAD)
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

	# Display the density map figure

	im_density = axs['d'].imshow(Local_Density_KDE, vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2, cmap='cividis')
	# Add a colorbar
	divider = make_axes_locatable(axs['d'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_density, cax=cax)
	axs['d'].set_title('Nuclei packing', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['d'].set_xticks([])
	axs['d'].set_yticks([])
	# Calculate the tick locations for Low, Medium, and High
	low_tick = 0
	high_tick = 1
	# Set ticks and labels for Low, Medium, and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])

	######################

	# # Display the area clustered blob labels figure

	im_area_cluster_labels = axs['e'].imshow(area_cluster_labels, alpha=ALPHA, cmap = 'brg')
	# Add a colorbar
	divider = make_axes_locatable(axs['e'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_area_cluster_labels, cax=cax)
	axs['e'].set_title(str(area_cluster_number) + ' nuclei groups by Area', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['e'].set_xticks([])
	axs['e'].set_yticks([])
	
	# Calculate the tick locations for Low, and High
	low_tick = 1
	high_tick = area_cluster_number
	# Set ticks and labels for Low and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])
	# cax.remove()

	######################

	# # Display the roundness clustered blob labels figure

	im_roundness_cluster_labels = axs['f'].imshow(roundness_cluster_labels, alpha=ALPHA, cmap = 'brg')
	# Add a colorbar
	divider = make_axes_locatable(axs['f'])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_roundness_cluster_labels, cax=cax)
	axs['f'].set_title(str(roundness_cluster_number) + ' nuclei groups by Roundness', pad = title_PAD)
	# Turn off axis ticks and labels
	axs['f'].set_xticks([])
	axs['f'].set_yticks([])

	# Calculate the tick locations for Low and High
	low_tick = 1
	high_tick = roundness_cluster_number
	# Set ticks and labels for Low and High
	cb.set_ticks([low_tick, high_tick])
	cb.set_ticklabels(['Low', 'High'])
	# cax.remove()

	######################
	
	return fig

##########################################################################

import streamlit.components.v1 as components
import base64
import io
from typing import Union, Tuple
import requests
from PIL import Image
import numpy as np

def read_image_and_convert_to_base64(image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, int, int]:
	"""
	Reads an image in PIL Image, file path, or numpy array format and returns a base64-encoded string of the image
	in JPEG format, along with its width and height.

	Args:
		image: An image in PIL Image, file path, or numpy array format.

	Returns:
		A tuple containing:
		- base64_src (str): A base64-encoded string of the image in JPEG format.
		- width (int): The width of the image in pixels.
		- height (int): The height of the image in pixels.

	Raises:
		TypeError: If the input image is not of a recognized type.

	Assumes:
		This function assumes that the input image is a valid image in PIL Image, file path, or numpy array format.
		It also assumes that the necessary libraries such as Pillow and scikit-image are installed.

	"""
	# Set the maximum image size to None to allow reading of large images
	Image.MAX_IMAGE_PIXELS = None

	# If input image is PIL Image, convert it to RGB format
	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')

	# If input image is a file path, open it using requests library if it's a URL, otherwise use PIL Image's open function
	elif isinstance(image, str):
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
		except:
			# If opening image using requests library fails, try to use scikit-image library to read the image
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")

			# Read the image using scikit-image and convert it to a PIL Image
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")

	# If input image is a numpy array, create a PIL Image from it
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")

	# If input image is not of a recognized type, raise a TypeError
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	# Get the width and height of the image
	width, height = image_pil.size

	# Save the PIL Image as a JPEG image with maximum quality (100) and no subsampling
	in_mem_file = io.BytesIO()
	image_pil.save(in_mem_file, format="JPEG", subsampling=0, quality=100)

	# Encode the bytes of the JPEG image in base64 format
	img_bytes = in_mem_file.getvalue()
	image_str = base64.b64encode(img_bytes).decode("utf-8")

	# Create a base64-encoded string of the image in JPEG format
	base64_src = f"data:image/jpg;base64,{image_str}"

	# Return the base64-encoded string along with the width and height of the image
	return base64_src, width, height

######################################################

def image_comparison(
	img1: str,
	img2: str,
	label1: str,
	label2: str,
	width_value = 674,
	show_labels: bool=True,
	starting_position: int=50,
) -> components.html:
	"""
	Creates an HTML block containing an image comparison slider of two images.

	Args:
		img1 (str): A string representing the path or URL of the first image to be compared.
		img2 (str): A string representing the path or URL of the second image to be compared.
		label1 (str): A label to be displayed above the first image in the slider.
		label2 (str): A label to be displayed above the second image in the slider.
		width_value (int, optional): The maximum width of the slider in pixels. Defaults to 500.
		show_labels (bool, optional): Whether to show the labels above the images in the slider. Defaults to True.
		starting_position (int, optional): The starting position of the slider. Defaults to 50.

	Returns:
		A Dash HTML component that displays an image comparison slider.

	"""
		# Convert the input images to base64 format
	img1_base64, img1_width, img1_height = read_image_and_convert_to_base64(img1)
	img2_base64, img2_width, img2_height = read_image_and_convert_to_base64(img2)

	# Get the maximum width and height of the input images
	img_width = int(max(img1_width, img2_width))
	img_height = int(max(img1_height, img2_height))

	# Calculate the aspect ratio of the images
	h_to_w = img_height / img_width

	# Determine the height of the slider based on the width and aspect ratio
	if img_width < width_value:
		width = img_width
	else:
		width = width_value
	height = int(width * h_to_w)

	# Load CSS and JS for the slider
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# Create the HTML code for the slider
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1_base64}',
					label: '{label1}',
				}},
				{{
					src: '{img2_base64}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {str(show_labels).lower()},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: true,
			}});
		</script>
		"""

	# Create a Dash HTML component from the HTML code
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################
