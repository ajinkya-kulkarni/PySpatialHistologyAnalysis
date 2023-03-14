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

import tensorflow as tf

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##########################################################################

from modules import *
from compare_images import *

##########################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PySpatialAnalysis', page_icon = image_bytes, layout = "centered", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PySpatialAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

##########################################################################

# Title of the web app

st.title(':blue[Spatial analysis of H&E images using PySpatialAnalysisWSI using StarDist]')

st.markdown("")

##########################################################################

with st.form(key = 'form1', clear_on_submit = True):

	st.markdown(':blue[Upload an H&E image to be analyzed. Works best for images smaller than 1000x1000 pixels ]')

	uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	######################################################################

	submitted = st.form_submit_button('Analyze')
	
	######################################################################

	if uploaded_file is None:
		st.stop()

	######################################################################

	if submitted:

		try:

			rgb_image = read_image(uploaded_file)

			relabelled_image = perform_analysis(rgb_image)

			modified_labels = np.where(relabelled_image > 0, 255, relabelled_image)

			tf.reset_default_graph()

			# Convert grayscale image to RGB image
			modified_labels_rgb_image = 255 * np.ones((*modified_labels.shape, 3), dtype=np.uint8)
			# modified_labels_rgb_image[:, :, 0] = 255 - modified_labels
			# modified_labels_rgb_image[:, :, 1] = 255 - modified_labels
			# modified_labels_rgb_image[:, :, 2] = 255 - modified_labels

			# Replace black pixels with "tab:blue"
			black_pixels = np.where(modified_labels == 255)
			modified_labels_rgb_image[black_pixels[0], black_pixels[1], :] = (31, 119, 180)

			# Replace white pixels with "cosmic latte"
			white_pixels = np.where(modified_labels == 0)
			modified_labels_rgb_image[white_pixels[0], white_pixels[1], :] = (247, 234, 199)

			image_comparison(
			img1 = rgb_image,
			img2 = modified_labels_rgb_image,
			label1="Image",
			label2="Result",
			width = 674,
			in_memory = True, show_labels = True, make_responsive = True
			)

		except:

			ErrorMessage = st.error('Error with analyzing the image', icon = None)
			st.stop()

		######################################################################

		st.stop()

##########################################################################