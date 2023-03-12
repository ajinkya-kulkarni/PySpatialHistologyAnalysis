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

import cv2
import os
import skimage
from tqdm.notebook import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

##########################################################################

def make_analysis(filename, grayscale=True, hsv=True):

	# Check if file exists
	if not os.path.isfile(filename):
		raise FileNotFoundError(f"File not found: {filename}")

	rgb_image = skimage.io.imread(filename, as_gray=False)
	
	# Normalize pixel values to 0-255 range and convert to 8-bit unsigned integer array
	rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	# Check if the last dimension is 3 and not 4
	if rgb_image.shape[-1] != 3:
		raise Exception ('RGB image has the wrong dimenion(s)')

	#########

	# Convert RGB color image to grayscale image
	if grayscale:
		gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
	else:
		gray_img = None
	
	# Convert RGB color image to HSV color image
	if hsv:
		hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
	else:
		hsv_image = None

	#########

	# # prints a list of available models
	# StarDist2D.from_pretrained()

	# creates a pretrained model
	model = StarDist2D.from_pretrained('2D_versatile_he')

	labels, more_info = model.predict_instances(normalize(rgb_image))

	rendered_labels = render_label(labels)

	return rgb_image, labels, more_info, rendered_labels

##########################################################################
