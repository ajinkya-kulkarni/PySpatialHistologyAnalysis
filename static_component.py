import streamlit.components.v1 as components
import base64
import io
from typing import Union, Tuple
import requests
from PIL import Image
import numpy as np

######################################################

def read_image_and_convert_to_base64(image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, int, int]:
	Image.MAX_IMAGE_PIXELS = None
	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')
	elif isinstance(image, str):
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
		except:
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	width, height = image_pil.size
	in_mem_file = io.BytesIO()
	image_pil.save(in_mem_file, format="JPEG", subsampling=0, quality=100)
	img_bytes = in_mem_file.getvalue()
	image_str = base64.b64encode(img_bytes).decode("utf-8")
	base64_src = f"data:image/jpg;base64,{image_str}"
	return base64_src, width, height

######################################################

def image_comparison(
	img1: str,
	img2: str,
	label1: str = "Before",
	label2: str = "After",
	show_labels: bool = True,
	starting_position: int = 50,
	make_responsive: bool = True
) -> components.html:

	img1_base64, img1_width, img1_height = read_image_and_convert_to_base64(img1)
	img2_base64, img2_width, img2_height = read_image_and_convert_to_base64(img2)

	img_width = int(max(img1_width, img2_width))
	img_height = int(max(img1_height, img2_height))

	h_to_w = img_height / img_width

	width = 500
	height = int((width * h_to_w) * 0.95)

	# Load CSS and JS
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# write html block
	htmlcode = f"""
		<style>
			body {{ margin: unset; }}
			#foo {{ margin: 0 auto; }}
		</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width or '100%'};"></div>
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
					showLabels: {'true'},
					showCredits: true,
					startingPosition: "{starting_position}%",
					makeResponsive: {'true'},
				}});
		</script>
		"""
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################