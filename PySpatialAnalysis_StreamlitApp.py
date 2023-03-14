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
from io import BytesIO

from skimage import measure

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##########################################################################

from modules import *

from compare_images import read_image_as_pil
from compare_images import pillow_to_base64
from compare_images import local_file_to_base64
from compare_images import pillow_local_file_to_base64
from compare_images import image_comparison

##########################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PySpatialHistologyAnalysis', page_icon = image_bytes, layout = "centered", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PySpatialHistologyAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

##########################################################################

# Title of the web app

st.title(':blue[Spatial analysis of H&E images using PySpatialHistologyAnalysis and StarDist]')

st.markdown("")

##########################################################################

with st.form(key = 'form1', clear_on_submit = True):

	st.markdown(':blue[Upload an H&E image/slide to be analyzed. Works best for images/slides smaller than 1000x1000 pixels ]')

	uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	######################################################################

	submitted = st.form_submit_button('Analyze')

	st.markdown("")
	
	######################################################################

	if uploaded_file is None:
		st.stop()

	######################################################################

	if submitted:

		with st.spinner('Analyzing uploaded image...'):

			rgb_image = read_image(uploaded_file)

			##########################################################

			labels, detailed_info = perform_analysis(rgb_image)

			##########################################################

			modified_labels = np.where(labels > 0, 255, labels)

			##########################################################

			# st.write(detailed_info)

			# Convert grayscale image to RGB image
			modified_labels_rgb_image = 255 * np.ones((*modified_labels.shape, 3), dtype=np.uint8)

			##########################################################

			# Replace black pixels with "tab:blue"
			black_pixels = np.where(modified_labels == 255)
			modified_labels_rgb_image[black_pixels[0], black_pixels[1], :] = (31, 119, 180)

			# Replace white pixels with custom RGB color
			white_pixels = np.where(modified_labels == 0)
			modified_labels_rgb_image[white_pixels[0], white_pixels[1], :] = (247, 234, 199)

		##############################################################

		# Check that each label is a unique integer
		unique_labels = np.unique(labels)
		num_labels = len(unique_labels) - 1  # subtract 1 to exclude background label
		if num_labels != labels.max():
			raise Exception ('Each blob does not have a unique integer assigned to it.')

		##############################################################
		
		image_comparison(img1 = rgb_image, img2 = modified_labels_rgb_image, label1="Uploaded image", label2="Result", width = 674, in_memory = True, show_labels = True, make_responsive = True)

		##############################################################

		# Compute the region properties
		label_properties = measure.regionprops_table(labels, intensity_image = rgb_image, properties=('area', 'centroid', 'eccentricity', 'equivalent_diameter','label', 'orientation', 'perimeter'))

		# Create a Pandas DataFrame
		dataframe = pd.DataFrame(label_properties)

		##############################################################

		centroids = list(zip(dataframe['centroid-0'], dataframe['centroid-1']))

		centroids = np.asarray(centroids)

		##############################################################

		st.markdown("")

		# Compute KDE heatmap
		kde_heatmap = compute_kde_heatmap(centroids, labels, subsample_factor = 5)

		##################################################################

		st.markdown("")

		##################################################################

		eccentricity_list = list(dataframe['eccentricity'])
		eccentricity_list = np.atleast_2d(np.asarray(eccentricity_list))

		cluster_number = 3

		# Cluster the labels by eccentricity
		cluster_labels = cluster_labels_by_eccentricity(eccentricity_list, labels, n_clusters=cluster_number)

		##################################################################

		# Create the figure and axis objects
		fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

		# Overlay the labels image and KDE heatmap on the first subplot
		im = axs[0].imshow(modified_labels, cmap = 'binary')
		im_heatmap = axs[0].imshow(kde_heatmap / kde_heatmap.max(), cmap='coolwarm', vmin = 0, vmax = 1, alpha=0.8)
		# Add a colorbar
		divider = make_axes_locatable(axs[0])
		cax = divider.append_axes("right", size="3%", pad=0.07)
		fig.colorbar(im_heatmap, cax=cax)
		axs[0].set_title('Kernel Density Estimate heatmap of Nuclei')
		# Turn off axis ticks and labels
		axs[0].set_xticks([])
		axs[0].set_yticks([])

		# Overlay the clustered blob labels on the second subplot
		im = axs[1].imshow(cluster_labels, cmap='viridis')
		# Add a colorbar
		divider = make_axes_locatable(axs[1])
		cax = divider.append_axes("right", size="3%", pad=0.07)
		fig.colorbar(im, cax=cax)
		axs[1].set_title('Nuclei colored by Eccentricity')
		# Turn off axis ticks and labels
		axs[1].set_xticks([])
		axs[1].set_yticks([])

		# Adjust the layout of the subplots
		plt.tight_layout()
		st.pyplot(fig)

		##################################################################

		st.markdown("")

		st.markdown("Detailed Report")

		# Rename some columns
		dataframe_renamed = dataframe.rename(columns={'area': 'Region Area', 'centroid': 'Region Centroid', 'eccentricity':'Eccentricity', 'equivalent_diameter':'Equivalent Diameter','orientation':'Orientation', 'label':'Label #'})

		st.dataframe(dataframe_renamed.style.format("{:.2f}"), use_container_width = True)

		##################################################################

		st.stop()

##########################################################################
