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

# UX/UI recommendations provided by Radhika Bhagwat (radhika.bhagwat3@gmail.com, Product Designer)

########################################################################################

import streamlit as st
from streamlit_image_comparison import image_comparison

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stardist.models import StarDist2D
from stardist.plot import render_label
from stardist import relabel_image_stardist
from csbdeep.utils import normalize

import numpy as np
from io import BytesIO

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

########################################################################################

from modules import *

########################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PySpatialAnalysis', page_icon = image_bytes, layout = "centered", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PySpatialAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

########################################################################################

# Title of the web app

st.title(':blue[Spatial analysis of H&E images using PySpatialAnalysisWSI using StarDist]')

st.markdown("")

########################################################################################

with st.form(key = 'form1', clear_on_submit = True):

	st.markdown(':blue[Upload an H&E image to be analyzed.]')

	uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	####################################################################################

	submitted = st.form_submit_button('Analyze')

	st.markdown("")
	st.markdown("")
	
	####################################################################################

	if uploaded_file is None:
		st.stop()

	####################################################################################

	if submitted:

		try:

			rgb_image = read_image(uploaded_file)

			# reads the H&E pretrained model
			model = StarDist2D.from_pretrained('2D_versatile_he')
			labels, more_info = model.predict_instances(normalize(rgb_image))
			rendered_labels = render_label(labels)

			relabelled_image = relabel_image_stardist(labels, n_rays = 128)
			modified_labels = np.where(relabelled_image > 0, 255, relabelled_image)

			modified_labels_rgb_image = np.empty((*modified_labels.shape, 3), dtype=np.uint8)
			# Set the values of the RGB channels to the grayscale values
			modified_labels_rgb_image[:, :, 0] = 255 - modified_labels
			modified_labels_rgb_image[:, :, 1] = 255 - modified_labels
			modified_labels_rgb_image[:, :, 2] = 255 - modified_labels

			# modified_labels_rgb_image = resize_image_by_width(modified_labels_rgb_image, 670)
			# rgb_image = resize_image_by_width(rgb_image, 670)

			# modified_labels_rgb_image = np.stack((modified_labels,)*3, axis=-1)

			image_comparison(
			img1 = rgb_image,
			img2 = modified_labels_rgb_image,
			label1="H&E Image",
			label2="Segmented H&E Image",
			width = 672
			)

		except:

			ErrorMessage = st.error('Error with analyzing the image', icon = None)
			st.stop()

		################################################################################

		st.stop()

################################################################################