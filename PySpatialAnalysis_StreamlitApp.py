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

import networkx as nx

from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib

from scipy.spatial import voronoi_plot_2d

import time

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

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
		'About': 'This is an application for demonstrating the PySpatialHistologyAnalysis package. Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen.'
	}
)

##########################################################################

# Set the title of the web app
st.title(':blue[Spatial analysis of H&E images]')

st.caption('Application screenshots available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis). Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/TestImage.jpeg). Source code available [here](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis).', unsafe_allow_html = False)

# Add some vertical space between the title and the next section
st.markdown("")

##########################################################################

# Create a form using the "form" method of Streamlit
with st.form(key = 'form1', clear_on_submit = True):

	# Add some text explaining what the user should do next
	st.markdown(':blue[Upload an H&E image/slide to be analyzed. Works best for images/slides smaller than 1000x1000 pixels]')

	# Add a file uploader to allow the user to upload an image file
	uploaded_file = st.file_uploader("Upload a file", type = ["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	######################################################################

	st.markdown("")

	left_column, middle_column, right_column = st.columns(3)

	with left_column:

		st.slider('Threshold (σ) for Nuclei detection. Higher value detects lesser Nuclei.', min_value = 0.1, max_value = 0.9, value = 0.5, step = 0.1, format = '%0.1f', label_visibility = "visible", key = '-SensitivityKey-')

		ModelSensitivity = round(float(st.session_state['-SensitivityKey-']), 2)

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

		st.markdown("")

		ProgressBarText = st.empty()
		ProgressBarText.caption("Analyzing uploaded image...")
		ProgressBar = st.progress(0)
		ProgressBarTime = 0.1

		# Read in the RGB image from an uploaded file
		rgb_image = read_image(uploaded_file)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(1/7))

		##########################################################

		# Normalize staining

		stain_normalized_rgb_image = normalize_staining(rgb_image)

		##########################################################
		
# 		perform_analysis_image = stain_normalized_rgb_image
		
		perform_analysis_image = rgb_image
		
		##########################################################

		# Perform instance segmentation analysis on the RGB image to obtain the labels
		# and detailed information about each label
		labelled_image, detailed_info = perform_analysis(perform_analysis_image, ModelSensitivity)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(2/7))

		##########################################################

		# Make RGB image from labels image

		modified_labels_rgb_image = colorize_labels(labelled_image)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(3/7))

		##############################################################

		# Check that each label is a unique integer
		unique_labels = np.unique(labelled_image)
		num_labels = len(unique_labels) - 1  # subtract 1 to exclude the background label
		if num_labels != labelled_image.max():
			raise Exception('Each blob does not have a unique integer assigned to it.')

		##############################################################

		# Compute the region properties for each label in the label image
		# using a function called "measure.regionprops_table"
		# The properties computed include area, centroid, label, and orientation
		label_properties = measure.regionprops_table(labelled_image, intensity_image=perform_analysis_image, properties=('area', 'axis_major_length', 'axis_minor_length', 'centroid', 'label', 'orientation'))

		# Create a Pandas DataFrame to store the region properties
		dataframe = pd.DataFrame(label_properties)

		axis_major_length = label_properties['axis_major_length']
		axis_minor_length = label_properties['axis_minor_length']

		roundness = (axis_minor_length / axis_major_length)

		dataframe['Roundness'] = roundness

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(4/7))

		##############################################################

		## Calculate local nuclei density

		Local_Density_mean_filter = mean_filter(labelled_image)

		Local_Density_mean_filter = normalize_density_maps(Local_Density_mean_filter)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(5/7))

		######

		Local_Density_KDE = weighted_kde_density_map(labelled_image, num_points = 1000)
		
		Local_Density_KDE = normalize_density_maps(Local_Density_KDE)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(6/7))

		##############################################################

		## Perform binning of data into clusters

		label_list = list(dataframe['label'])

		area_cluster_labels = bin_property_values(labelled_image, list(dataframe['area']), area_cluster_number)

		roundness_cluster_labels = bin_property_values(labelled_image, list(dataframe['Roundness']), roundness_cluster_number)

		time.sleep(ProgressBarTime)
		ProgressBar.progress(float(7/7))

		ProgressBarText.empty()
		ProgressBar.empty()

		##############################################################

		st.markdown("""---""")

		st.markdown("Results")

		##############################################################

		## Generate visualizations of the uploaded RGB image and the results of the instance segmentation analysis
		## using a function called "make_plots"

		# result_figure = make_first_plot(rgb_image, stain_normalized_rgb_image)

		# ## Display the figure using Streamlit's "st.pyplot" function
		# st.pyplot(result_figure)

		image_comparison(img1=rgb_image, img2=stain_normalized_rgb_image, label1="Uploaded H&E image", label2="Stain normalized H&E image")

		st.markdown("""---""")

		##############################################################

		# Compare the uploaded RGB image with the modified label image
		# using a function called "image_comparison"
		# Set parameters for image width, in-memory display, and responsiveness

		image_comparison(img1=perform_analysis_image, img2=modified_labels_rgb_image, label1="Uploaded H&E image", label2="Segmented H&E image")

		st.markdown("""---""")

		##############################################################

		## Generate visualizations of the uploaded RGB image and the results of the instance segmentation analysis
		## using a function called "make_plots"

		result_figure = make_second_plot(perform_analysis_image, ModelSensitivity, modified_labels_rgb_image, detailed_info, Local_Density_mean_filter, Local_Density_KDE, area_cluster_labels, area_cluster_number, roundness_cluster_labels, roundness_cluster_number)

		## Display the figure using Streamlit's "st.pyplot" function
		st.pyplot(result_figure)

		# # Save the figure to a file
		# save_filename = 'Result_' + uploaded_file.name[:-4] + '.png'
		# result_figure.savefig(save_filename, bbox_inches='tight')
		# # Close the figure
		# plt.close(result_figure)

		##################################################################

		with st.spinner('Generating Nuclei connectivity graphs...'):

			# Call the make_graph function to get the graph and node labels
			graph, labels = make_network_connectivity_graph(labelled_image)

			# Compute Voronoi tessellation of the labelled image
			vor = voronoi_tessellation(labelled_image)

		##################################################################

		st.markdown("""---""")

		st.markdown("Voronoi Tesselation, indicating Nuclei packing")

		fig, ax = plt.subplots()

		# Add the labels image with transparency
		plt.imshow(perform_analysis_image, alpha = 0.5)

		# Find the limits of the image
		ymax, xmax = labelled_image.shape

		# Plot the Voronoi diagram
		voronoi_plot_2d(vor, ax=ax, show_vertices = False, line_colors = 'k', show_points = False, line_width = 0.5)

		# Set the limits of the plot to match the original image
		ax.set_xlim([0, xmax])
		ax.set_ylim([0, ymax])

		ax.set_xticks([])
		ax.set_yticks([])

		st.pyplot(fig)
		
		##################################################################

		st.markdown("""---""")

		st.markdown("Nuclei connectivity graph, indicating similar spaced Nuclei clusters")

		# Define node colors
		unique_labels = np.unique(labels)
		num_colors = len(unique_labels)
		base_cmap = matplotlib.colormaps['Set1']
		cmap = ListedColormap(base_cmap(np.linspace(0, 1, num_colors)))
		node_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

		# Create a figure and axis
		fig, ax = plt.subplots()

		# Draw the graph
		pos = nx.get_node_attributes(graph, 'pos')
		nx.draw_networkx_nodes(graph, pos, node_color=[node_colors[label] for label in labels], node_size = 5, ax=ax)
		nx.draw_networkx_edges(graph, pos, edge_color='gray', width = 1, ax=ax)

		# Add the labels image with transparency
		plt.imshow(perform_analysis_image, alpha=0.5)

		st.pyplot(fig)

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
