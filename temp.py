def image_comparison(
	img1: str,
	img2: str,
	label1: str = "1",
	label2: str = "2",
	starting_position: int = 50,
) -> components.html:

	# Prepare images
	img1_pillow = read_image_as_pil(img1)
	img2_pillow = read_image_as_pil(img2)

	img_width, img_height = img1_pillow.size
	h_to_w = img_height / img_width
	width = 674
	height = int((width * h_to_w) * 0.95)

    # Convert images to base64 strings
    img1 = pillow_to_base64(img1_pillow)
    img2 = pillow_to_base64(img2_pillow)

	# Load CSS and JS
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# write html block
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width or '100%'};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1}',
					label: '{label1}',
				}},
				{{
					src: '{img2}',
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