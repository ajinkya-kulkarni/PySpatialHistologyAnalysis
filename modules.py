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

def make_plots(rgb_image, detailed_info, modified_labels_rgb_image, modified_labels, Local_Density, kde_heatmap, criterion, cluster_labels, cluster_number, SIZE = "3%", PAD = 0.08, title_PAD = 10, DPI = 300, ALPHA = 1):

	gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

	# Create the figure and axis objects
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi = DPI)

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
	
	# Display labelled image
	im = axs[0, 1].imshow(modified_labels_rgb_image)
	# Add a colorbar
	divider = make_axes_locatable(axs[0, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im, cax=cax)
	axs[0, 1].set_title('Segmented Nuclei, N=' + str(len(detailed_info['points'])), pad = title_PAD)
	# Turn off axis ticks and labels
	axs[0, 1].set_xticks([])
	axs[0, 1].set_yticks([])
	cax.remove()

	# # Overlay the grayscale image and KDE heatmap
	# # im = axs[1, 0].imshow(gray_image, cmap = 'gray_r', zorder = 1)
	# im_heatmap = axs[1, 0].imshow(kde_heatmap / kde_heatmap.max(), cmap='jet', vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2)
	# # Add a colorbar
	# divider = make_axes_locatable(axs[1, 0])
	# cax = divider.append_axes("right", size=SIZE, pad=PAD)
	# cb = fig.colorbar(im_heatmap, cax=cax)
	# axs[1, 0].set_title('Nuclei clusters', pad = title_PAD)
	# # Turn off axis ticks and labels
	# axs[1, 0].set_xticks([])
	# axs[1, 0].set_yticks([])

	# Display the clustered blob labels figure
	im_clusters = axs[1, 0].imshow(cluster_labels, alpha=ALPHA, cmap='viridis')
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 0])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_clusters, cax=cax)
	axs[1, 0].set_title(str(cluster_number - 1) + ' nuclei groups by ' + criterion, pad = title_PAD)
	# Turn off axis ticks and labels
	axs[1, 0].set_xticks([])
	axs[1, 0].set_yticks([])
	# get tick locations and update to integers
	cb.set_ticks(range(cluster_number))
	tick_locs = cb.get_ticks()
	int_tick_labels = [int(tick) for tick in tick_locs]
	cb.set_ticklabels(int_tick_labels)

	# Display the density map figure
	# im = axs[2, 0].imshow(gray_image, cmap = 'gray_r', zorder = 1)
	im_density = axs[1, 1].imshow(Local_Density, vmin = 0, vmax = 1, alpha=ALPHA, zorder = 2, cmap='Spectral_r')
	# Add a colorbar
	divider = make_axes_locatable(axs[1, 1])
	cax = divider.append_axes("right", size=SIZE, pad=PAD)
	cb = fig.colorbar(im_density, cax=cax)
	axs[1, 1].set_title('Local Nuclei Density', pad = title_PAD)
	# Turn off axis ticks and labels
	axs[1, 1].set_xticks([])
	axs[1, 1].set_yticks([])

	# # Remove the last subplot in the bottom row
	# fig.delaxes(axs[2, 1])
	
	return fig

##########################################################################

import streamlit.components.v1 as components
import base64
import io
import uuid
from typing import Union
import requests

TEMP_DIR = "temp"

def exif_transpose(image: Image.Image):
	"""
	Transpose a PIL image accordingly if it has an EXIF Orientation tag.
	Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()
	:param image: The image to transpose.
	:return: An image.
	"""
	exif = image.getexif()
	orientation = exif.get(0x0112, 1)  # default 1
	if orientation > 1:
		method = {
			2: Image.FLIP_LEFT_RIGHT,
			3: Image.ROTATE_180,
			4: Image.FLIP_TOP_BOTTOM,
			5: Image.TRANSPOSE,
			6: Image.ROTATE_270,
			7: Image.TRANSVERSE,
			8: Image.ROTATE_90,
		}.get(orientation)
		if method is not None:
			image = image.transpose(method)
			del exif[0x0112]
			image.info["exif"] = exif.tobytes()
	return image

def read_image_as_pil(image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False):
	"""
	Loads an image as PIL.Image.Image.
	Args:
		image : Can be image path or url (str), numpy image (np.ndarray) or PIL.Image
	"""
	# https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
	Image.MAX_IMAGE_PIXELS = None

	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')
	elif isinstance(image, str):
		# read image if str image path is provided
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
			if exif_fix:
				image_pil = exif_transpose(image_pil)
		except:  # handle large/tiff image reading
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:  # b&w
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:  # rgba
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:  # rgb
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:  # image in CHW
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	return image_pil

def pillow_to_base64(image: Image.Image) -> str:
	"""
	Convert a PIL image to a base64-encoded string.

	Parameters
	----------
	image: PIL.Image.Image
		The image to be converted.

	Returns
	-------
	str
		The base64-encoded string.
	"""
	in_mem_file = io.BytesIO()
	image.save(in_mem_file, format="JPEG", subsampling=0, quality=100)
	img_bytes = in_mem_file.getvalue()  # bytes
	image_str = base64.b64encode(img_bytes).decode("utf-8")
	base64_src = f"data:image/jpg;base64,{image_str}"
	return base64_src

def local_file_to_base64(image_path: str) -> str:
	"""
	Convert a local image file to a base64-encoded string.

	Parameters
	----------
	image_path: str
		The path to the image file.

	Returns
	-------
	str
		The base64-encoded string.
	"""
	file_ = open(image_path, "rb")
	img_bytes = file_.read()
	image_str = base64.b64encode(img_bytes).decode("utf-8")
	file_.close()
	base64_src = f"data:image/jpg;base64,{image_str}"
	return base64_src

def pillow_local_file_to_base64(image: Image.Image, temp_dir: str):
	"""
	Convert a Pillow image to a base64 string, using a temporary file on disk.

	Parameters
	----------
	image : PIL.Image.Image
		The Pillow image to convert.
	temp_dir : str
		The directory to use for the temporary file.

	Returns
	-------
	str
		A base64-encoded string representing the image.
	"""
	# Create temporary file path using os.path.join()
	img_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".jpg")

	# Save image to temporary file
	image.save(img_path, subsampling=0, quality=100)

	# Convert temporary file to base64 string
	base64_src = local_file_to_base64(img_path)

	return base64_src

def image_comparison(
	img1: str,
	img2: str,
	label1: str = "1",
	label2: str = "2",
	show_labels: bool = True,
	starting_position: int = 50,
	make_responsive: bool = True,
	in_memory: bool = False,
) -> components.html:
	"""
	Create a comparison slider for two images.
	
	Parameters
	----------
	img1: str
		Path to the first image.
	img2: str
		Path to the second image.
	label1: str, optional
		Label for the first image. Default is "1".
	label2: str, optional
		Label for the second image. Default is "2".
	width: int, optional
		Width of the component in pixels. Default is 674.
	show_labels: bool, optional
		Whether to show labels on the images. Default is True.
	starting_position: int, optional
		Starting position of the slider as a percentage (0-100). Default is 50.
	make_responsive: bool, optional
		Whether to enable responsive mode. Default is True.
	in_memory: bool, optional
		Whether to handle pillow to base64 conversion in memory without saving to local. Default is False.

	Returns
	-------
	components.html
		Returns a static component with a timeline
	"""
	# Prepare images
	img1_pillow = read_image_as_pil(img1)
	img2_pillow = read_image_as_pil(img2)

	img_width, img_height = img1_pillow.size
	h_to_w = img_height / img_width
	width = 674
	height = int((width * h_to_w) * 0.95)

	if in_memory:
		# Convert images to base64 strings
		img1 = pillow_to_base64(img1_pillow)
		img2 = pillow_to_base64(img2_pillow)
	else:
		# Create base64 strings from temporary files
		os.makedirs(TEMP_DIR, exist_ok=True)
		for file_ in os.listdir(TEMP_DIR):
			if file_.endswith(".jpg"):
				os.remove(os.path.join(TEMP_DIR, file_))
		img1 = pillow_local_file_to_base64(img1_pillow, TEMP_DIR)
		img2 = pillow_local_file_to_base64(img2_pillow, TEMP_DIR)

	# Load CSS and JS
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# write html block
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width or '100%'};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1}',
					label: '{label1}',
				}},
				{{
					src: '{img2}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {'true' if show_labels else 'false'},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: {'true' if make_responsive else 'false'},
			}});
		</script>
		"""
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################
