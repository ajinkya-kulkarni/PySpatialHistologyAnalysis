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

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##########################################################################

from modules import *

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

		st.markdown("""---""")

		st.markdown("Results")

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

		st.markdown("")

		##############################################################

		# Compute the region properties
		label_properties = measure.regionprops_table(labels, intensity_image = rgb_image, properties=('area', 'centroid', 'eccentricity','label', 'orientation'))

		# Create a Pandas DataFrame
		dataframe = pd.DataFrame(label_properties)

		##############################################################

		centroids = list(zip(dataframe['centroid-0'], dataframe['centroid-1']))

		centroids = np.asarray(centroids)

		##############################################################

		with st.spinner('Creating plots and report...'):

			# Compute KDE heatmap

			subsample_factor = 2

			kde_heatmap = compute_kde_heatmap(centroids, labels, subsample_factor)

			##############################################################

			# Choose criterion from ['area', 'eccentricity', 'orientation']

			criterion = 'eccentricity'

			criterion_list = list(dataframe[criterion])
			criterion_list = np.atleast_2d(np.asarray(criterion_list))

			cluster_number = 4

			# Cluster the labels by criterion
			cluster_labels = cluster_labels_by_criterion(criterion_list, labels, n_clusters=cluster_number)

			##############################################################

			figure = make_plots(rgb_image, modified_labels_rgb_image, modified_labels, kde_heatmap, criterion, cluster_labels, cluster_number)

			st.pyplot(figure)

		##################################################################

		st.markdown("""---""")

		# with st.spinner('Creating report...'):

		# Rename some columns
		dataframe_renamed = dataframe.rename(columns={'area': 'Region Area', 'centroid': 'Region Centroid', 'eccentricity':'Eccentricity', 'equivalent_diameter':'Equivalent Diameter','orientation':'Orientation', 'label':'Label #'})

		dataframe_renamed = dataframe_renamed.drop(['centroid-0', 'centroid-1'], axis=1)

		# remove the 'label' column and save it to a variable
		label_col = dataframe_renamed.pop('Label #')

		# insert the 'label' column back into the dataframe as the 1st column
		dataframe_renamed.insert(0, 'Label #', label_col)

		dataframe_renamed['Label #'] = dataframe_renamed['Label #'].astype(int)

		dataframe_renamed['Orientation'] = np.rad2deg(dataframe_renamed['Orientation']) + 90

		# BlankIndex = [''] * len(dataframe_renamed)
		# dataframe_renamed.index = BlankIndex

		st.markdown("Detailed Report")

		st.dataframe(dataframe_renamed.style.format("{:.2f}"), use_container_width = True)

		##################################################################

		st.stop()

##########################################################################
