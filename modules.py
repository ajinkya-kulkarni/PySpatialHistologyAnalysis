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

##########################################################################

def read_image(filename):

	# Load the image using Pillow and convert to 8 bit
	img = Image.open(filename)

	rgb_image = img.convert('RGB')

	# Convert the image to a numpy array
	rgb_image = np.array(rgb_image)

	return rgb_image

##########################################################################

def resize_image_by_width(image_path, target_width):
	
	# Get original dimensions
	height, width, _ = img.shape

	# Calculate scale factor to resize width to target_width
	scale_factor = target_width / width

	# Resize image with calculated scale factor
	resized_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))

	return resized_img

##########################################################################