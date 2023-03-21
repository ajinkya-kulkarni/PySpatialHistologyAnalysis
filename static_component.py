import streamlit.components.v1 as components
import base64
import io
from typing import Union, Tuple
import requests
from PIL import Image
import numpy as np

######################################################

def read_image_and_convert_to_base64(image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, int, int]:
	"""
	Reads an image in PIL Image, file path, or numpy array format and returns a base64-encoded string of the image
	in JPEG format, along with its width and height.

	Args:
		image: An image in PIL Image, file path, or numpy array format.

	Returns:
		A tuple containing:
		- base64_src (str): A base64-encoded string of the image in JPEG format.
		- width (int): The width of the image in pixels.
		- height (int): The height of the image in pixels.

	Raises:
		TypeError: If the input image is not of a recognized type.

	Assumes:
		This function assumes that the input image is a valid image in PIL Image, file path, or numpy array format.
		It also assumes that the necessary libraries such as Pillow and scikit-image are installed.

	"""
	# Set the maximum image size to None to allow reading of large images
	Image.MAX_IMAGE_PIXELS = None

	# If input image is PIL Image, convert it to RGB format
	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')

	# If input image is a file path, open it using requests library if it's a URL, otherwise use PIL Image's open function
	elif isinstance(image, str):
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
		except:
			# If opening image using requests library fails, try to use scikit-image library to read the image
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")

			# Read the image using scikit-image and convert it to a PIL Image
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")

	# If input image is a numpy array, create a PIL Image from it
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")

	# If input image is not of a recognized type, raise a TypeError
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	# Get the width and height of the image
	width, height = image_pil.size

	# Save the PIL Image as a JPEG image with maximum quality (100) and no subsampling
	in_mem_file = io.BytesIO()
	image_pil.save(in_mem_file, format="JPEG", subsampling=0, quality=100)

	# Encode the bytes of the JPEG image in base64 format
	img_bytes = in_mem_file.getvalue()
	image_str = base64.b64encode(img_bytes).decode("utf-8")

	# Create a base64-encoded string of the image in JPEG format
	base64_src = f"data:image/jpg;base64,{image_str}"

	# Return the base64-encoded string along with the width and height of the image
	return base64_src, width, height

######################################################

def image_comparison(
	img1: str,
	img2: str,
	label1: str,
	label2: str,
	width_value = 1169,
	show_labels: bool=True,
	starting_position: int=50,
) -> components.html:
	"""
	Creates an HTML block containing an image comparison slider of two images.

	Args:
		img1 (str): A string representing the path or URL of the first image to be compared.
		img2 (str): A string representing the path or URL of the second image to be compared.
		label1 (str): A label to be displayed above the first image in the slider.
		label2 (str): A label to be displayed above the second image in the slider.
		width_value (int, optional): The maximum width of the slider in pixels. Defaults to 500.
		show_labels (bool, optional): Whether to show the labels above the images in the slider. Defaults to True.
		starting_position (int, optional): The starting position of the slider. Defaults to 50.

	Returns:
		A Dash HTML component that displays an image comparison slider.

	"""
		# Convert the input images to base64 format
	img1_base64, img1_width, img1_height = read_image_and_convert_to_base64(img1)
	img2_base64, img2_width, img2_height = read_image_and_convert_to_base64(img2)

	# Get the maximum width and height of the input images
	img_width = int(max(img1_width, img2_width))
	img_height = int(max(img1_height, img2_height))

	# Calculate the aspect ratio of the images
	h_to_w = img_height / img_width

	# Determine the height of the slider based on the width and aspect ratio
	if img_width < width_value:
		width = img_width
	else:
		width = width_value
	height = int(width * h_to_w)

	# Load CSS and JS for the slider
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# Create the HTML code for the slider
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1_base64}',
					label: '{label1}',
				}},
				{{
					src: '{img2_base64}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {str(show_labels).lower()},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: true,
			}});
		</script>
		"""

	# Create a Dash HTML component from the HTML code
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################
