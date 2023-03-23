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
from io import BytesIO
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##########################################################################

from modules import *
from static_component import *

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
st.title(':blue[Spatial analysis of H&E images using PySpatialHistologyAnalysis and StarDist]')

st.caption('For more information, have a look at [this screenshot](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot1.png) and [this screenshot](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot2.png). Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/TestImage.png). Source code available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis).', unsafe_allow_html = False)

# Add some vertical space between the title and the next section
st.markdown("")

##########################################################################

# Create a form using the "form" method of Streamlit
with st.form(key = 'form1', clear_on_submit = True):

	# Add some text explaining what the user should do next
	st.markdown(':blue[Upload an H&E image/slide to be analyzed. Works best for images/slides smaller than 1000x1000 pixels]')

	# Add a file uploader to allow the user to upload an image file
	uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility='collapsed')

	######################################################################

	st.markdown("")

	left_column, middle_column, right_column = st.columns(3)

	with left_column:

		st.number_input('Sensitivity for Nuclei detection. Higher values detect lesser Nuclei.', key = '-SensitivityKey-', min_value = 0.1, max_value = 0.9, value = 0.5, step = 0.1, format = '%0.1f')

		ModelSensitivity = round(float(st.session_state['-SensitivityKey-']), 1)

	with middle_column:

		st.number_input('Number of classes for Area, between 1 and 10.', key = '-n_clusters_area_key-', min_value = 1, max_value = 10, value = 3, step = 1, format = '%d')

		area_cluster_number = int(st.session_state['-n_clusters_area_key-'])

	with right_column:

		st.number_input('Number of classes for Roundness, between 1 and 10.', key = '-n_clusters_roundness_key-', min_value = 1, max_value = 10, value = 3, step = 1, format = '%d')

		roundness_cluster_number= int(st.session_state['-n_clusters_roundness_key-'])

	st.markdown("")

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
			labels, detailed_info = perform_analysis(rgb_image, ModelSensitivity)

			##########################################################

			# Make RGB image from labels image

			modified_labels_rgb_image = colorize_labels(labels)

			##############################################################

			# Check that each label is a unique integer
			unique_labels = np.unique(labels)
			num_labels = len(unique_labels) - 1  # subtract 1 to exclude the background label
			if num_labels != labels.max():
				raise Exception('Each blob does not have a unique integer assigned to it.')

			##############################################################

			# Compare the uploaded RGB image with the modified label image
			# using a function called "image_comparison"
			# Set parameters for image width, in-memory display, and responsiveness

			image_comparison(img1=rgb_image, img2=modified_labels_rgb_image, label1="Uploaded image", label2="Segmented image")

			# Add a markdown line break
			st.markdown("")

			##############################################################

			# Compute the region properties for each label in the label image
			# using a function called "measure.regionprops_table"
			# The properties computed include area, centroid, label, and orientation
			label_properties = measure.regionprops_table(labels, intensity_image=rgb_image, properties=('area', 'axis_major_length', 'axis_minor_length', 'centroid', 'label', 'orientation'))

			# Create a Pandas DataFrame to store the region properties
			dataframe = pd.DataFrame(label_properties)

			axis_major_length = label_properties['axis_major_length']
			axis_minor_length = label_properties['axis_minor_length']

			roundness = (axis_minor_length / axis_major_length)

			dataframe['Roundness'] = roundness

			##############################################################

			## Calculate local nuclei density

			# nuclei_density = count_nuclei_per_window(labels)
			# nuclei_density = nuclei_density / nuclei_density.max()
			# Local_Density = nuclei_density.copy()

			Local_Density = calculate_local_density(labels)

			##############################################################

			## Perform binning of data into clusters

			label_list = list(dataframe['label'])

			area_cluster_labels = bin_property_values(labels, list(dataframe['area']), area_cluster_number)

			roundness_cluster_labels = bin_property_values(labels, list(dataframe['Roundness']), roundness_cluster_number)

			##############################################################

			## Generate visualizations of the uploaded RGB image and the results of the instance segmentation analysis
			## using a function called "make_plots"

			result_figure = make_plots(modified_labels_rgb_image, detailed_info, Local_Density, area_cluster_labels, area_cluster_number, roundness_cluster_labels, roundness_cluster_number)

			## Display the figure using Streamlit's "st.pyplot" function
			st.pyplot(result_figure)

			# # Save the figure to a file
			# save_filename = 'Result_' + uploaded_file.name[:-4] + '.png'
			# result_figure.savefig(save_filename, bbox_inches='tight')
			# # Close the figure
			# plt.close(result_figure)

		##################################################################

		st.markdown("""---""")

		# Define a mapping of the old column names to the new column names
		column_mapping = {'area': 'Region Area', 'centroid-0': 'Region Centroid-0', 'centroid-1': 'Region Centroid-1', 'equivalent_diameter': 'Equivalent Diameter', 'orientation': 'Orientation', 'label': 'Label #'}

		# Rename the columns of the DataFrame using the mapping
		renamed_dataframe = dataframe.rename(columns=column_mapping)

		# Remove the 'Region Centroid-0' and 'Region Centroid-1' columns from the DataFrame
		renamed_dataframe = renamed_dataframe.drop(columns=['Region Centroid-0', 'Region Centroid-1'])

		renamed_dataframe = renamed_dataframe.drop(columns=['axis_major_length', 'axis_minor_length'])
		
		# Move the 'Label #' column to the beginning of the DataFrame
		cols = list(renamed_dataframe.columns)
		cols.pop(cols.index('Label #'))
		renamed_dataframe = renamed_dataframe[['Label #'] + cols]

		# Convert the 'Orientation' column from radians to degrees and shift by 90 degrees using the "apply" method
		renamed_dataframe['Orientation'] = np.rad2deg(renamed_dataframe['Orientation']).add(90)

		# Display the detailed report
		st.markdown("Detailed Report")

		# Show the dataframe
		st.dataframe(renamed_dataframe, use_container_width = True)

		##################################################################

		st.stop()

##########################################################################
