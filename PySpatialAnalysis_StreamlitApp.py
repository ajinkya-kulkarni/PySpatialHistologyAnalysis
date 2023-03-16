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

# Open the logo file in binary mode and read its contents into memory
with open("logo.jpg", "rb") as f:
	image_data = f.read()

# Create a BytesIO object from the image data
image_bytes = BytesIO(image_data)

# Configure the page settings using the "set_page_config" method of Streamlit
st.set_page_config(
	page_title='PySpatialHistologyAnalysis',
	page_icon=image_bytes,  # Use the logo image as the page icon
	layout="centered",
	initial_sidebar_state="expanded",
	menu_items={
		'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de',
		'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de',
		'About': 'This is an application for demonstrating the PySpatialHistologyAnalysis package. Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'
	}
)

##########################################################################

# Set the title of the web app
st.title(':blue[Spatial analysis of H&E images using PySpatialHistologyAnalysis, PySAL and StarDist]')

st.caption('For more information, have a look at [this screenshot](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot1.png) and [this screenshot](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot2.png). Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/TestImage.png). Source code available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis).', unsafe_allow_html = False)

# Add some vertical space between the title and the next section
st.markdown("")

##########################################################################

# Create a form using the "form" method of Streamlit
with st.form(key='form1', clear_on_submit=True):

	# Add some text explaining what the user should do next
	st.markdown(':blue[Upload an H&E image/slide to be analyzed. Works best for images/slides smaller than 1000x1000 pixels]')

	# Add a file uploader to allow the user to upload an image file
	uploaded_file = st.file_uploader(
		"Upload a file",
		type=["tif", "tiff", "png", "jpg", "jpeg"],
		accept_multiple_files=False,
		label_visibility='collapsed'
	)

	######################################################################

	# Add a submit button to the form
	submitted = st.form_submit_button('Analyze')

	######################################################################

	# If no file was uploaded, stop processing and exit early
	if uploaded_file is None:
		st.stop()

	######################################################################

	if submitted:

		st.markdown("""---""")

		st.markdown("Results")

		with st.spinner('Analyzing uploaded image...'):

			# Read in the RGB image from an uploaded file
			rgb_image = read_image(uploaded_file)

			##########################################################

			# Perform instance segmentation analysis on the RGB image to obtain the labels
			# and detailed information about each label
			labels, detailed_info = perform_analysis(rgb_image)

			##########################################################

			# Convert the label image to a binary image where non-zero pixels are set to 255
			modified_labels = np.where(labels > 0, 255, labels)

			##########################################################

			# Create an RGB image with the same dimensions as the modified label image
			# with all pixels set to white
			modified_labels_rgb_image = 255 * np.ones((*modified_labels.shape, 3), dtype=np.uint8)

			##########################################################

			# Replace black (255) pixels in the modified label image with a custom blue color
			black_pixels = np.where(modified_labels == 255)
			modified_labels_rgb_image[black_pixels[0], black_pixels[1], :] = (31, 119, 180)

			# Replace white (0) pixels in the modified label image with a custom RGB color
			white_pixels = np.where(modified_labels == 0)
			modified_labels_rgb_image[white_pixels[0], white_pixels[1], :] = (247, 234, 199)

		##############################################################

		# Check that each label is a unique integer
		unique_labels = np.unique(labels)
		num_labels = len(unique_labels) - 1  # subtract 1 to exclude background label
		if num_labels != labels.max():
			raise Exception('Each blob does not have a unique integer assigned to it.')

		##############################################################

		# Compare the uploaded RGB image with the modified label image
		# using a function called "image_comparison"
		# Set parameters for image width, in-memory display, and responsiveness
		image_comparison(img1=rgb_image, img2=modified_labels_rgb_image, label1="Uploaded image", label2="Result", in_memory=True, show_labels=True, make_responsive=True)

		# Add a markdown line break
		st.markdown("")

		##############################################################

		# Compute the region properties for each label in the label image
		# using a function called "measure.regionprops_table"
		# The properties computed include area, centroid, eccentricity, label, and orientation
		label_properties = measure.regionprops_table(labels, intensity_image=rgb_image, properties=('area', 'centroid', 'eccentricity','label', 'orientation'))

		# Create a Pandas DataFrame to store the region properties
		dataframe = pd.DataFrame(label_properties)

		##############################################################

		# Extract the centroid coordinates from the DataFrame and convert them to a NumPy array
		centroids = list(zip(dataframe['centroid-1'], dataframe['centroid-0']))
		centroids = np.asarray(centroids)

		##############################################################

		with st.spinner('Creating plots and report...'):

			# Define a subsampling factor for the KDE heatmap
			subsample_factor = 2

			# Compute a kernel density estimate (KDE) heatmap of the centroid coordinates
			# using a function called "compute_kde_heatmap"
			kde_heatmap = compute_kde_heatmap(centroids, labels, subsample_factor)

			# # Calculate Pairwise Distance Heatmap of Label Centroids
			# pairwise_distances = plot_neighborhood_analysis(labels)

			##############################################################

			# Choose a criterion to cluster the labels on
			# criterion = 'eccentricity'
			criterion = 'area'

			# Specify the number of clusters to use for KMeans clustering
			cluster_number = 4

			# Extract the values of the chosen criterion for each label and convert to a 2D NumPy array
			criterion_list = list(dataframe[criterion])
			criterion_list = np.atleast_2d(np.asarray(criterion_list))

			# Cluster the labels based on the chosen criterion using a function called "cluster_labels_by_criterion"
			cluster_labels = cluster_labels_by_criterion(criterion_list, labels, n_clusters=cluster_number)

			##############################################################

			# Generate visualizations of the uploaded RGB image and the results of the instance segmentation analysis
			# using a function called "make_plots"
			figure = make_plots(rgb_image, detailed_info, modified_labels_rgb_image, modified_labels, kde_heatmap, criterion, cluster_labels, cluster_number)

			# Display the figure using Streamlit's "st.pyplot" function
			st.pyplot(figure)

		##################################################################

		st.markdown("""---""")

		# Define a mapping of the old column names to the new column names
		column_mapping = {
			'area': 'Region Area',
			'centroid-0': 'Region Centroid-0',
			'centroid-1': 'Region Centroid-1',
			'eccentricity': 'Eccentricity',
			'equivalent_diameter': 'Equivalent Diameter',
			'orientation': 'Orientation',
			'label': 'Label #'
		}

		# Rename the columns of the DataFrame using the mapping
		renamed_dataframe = dataframe.rename(columns=column_mapping)

		# Remove the 'Region Centroid-0' and 'Region Centroid-1' columns from the DataFrame
		renamed_dataframe = renamed_dataframe.drop(columns=['Region Centroid-0', 'Region Centroid-1'])
		
		# Move the 'Label #' column to the beginning of the DataFrame
		cols = list(renamed_dataframe.columns)
		cols.pop(cols.index('Label #'))
		renamed_dataframe = renamed_dataframe[['Label #'] + cols]

		# Make all the columns except 'Eccentricity' integer type using the "astype" method
		int_columns = [c for c in renamed_dataframe.columns if c != 'Eccentricity' and c!= 'Orientation']
		renamed_dataframe[int_columns] = renamed_dataframe[int_columns].astype(int)

		# Convert the 'Orientation' column from radians to degrees and shift by 90 degrees using the "apply" method
		renamed_dataframe['Orientation'] = np.rad2deg(renamed_dataframe['Orientation']).add(90)

		# Display the detailed report
		st.markdown("Detailed Report")

		# Show the dataframe
		st.dataframe(renamed_dataframe, use_container_width = True)

		##################################################################

		st.stop()

##########################################################################
