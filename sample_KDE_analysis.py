import numpy as np
from skimage.measure import regionprops
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def weighted_kde_density_map(image, bandwidth='auto', kernel='gaussian'):
	# Extract nucleus centroids and areas
	regions = regionprops(image)
	centroids = np.array([region.centroid for region in regions])
	areas = np.array([region.area for region in regions])

	# Compute weighted KDE
	if bandwidth == 'auto':
		# Rule of thumb bandwidth selection
		n = centroids.shape[0]
		d = centroids.shape[1]
		bandwidth = n**(-1.0/(d+4)) * np.std(centroids, axis=0).mean()

	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(centroids, sample_weight=areas)

	# Create a density map
	x = np.arange(0, image.shape[1], 1)
	y = np.arange(0, image.shape[0], 1)
	xx, yy = np.meshgrid(x, y)

	coords = np.vstack([yy.ravel(), xx.ravel()]).T
	density = np.exp(kde.score_samples(coords))
	density_map = density.reshape(image.shape)

	return density_map

# Load or create your labeled image (replace 'labeled_image' with your actual labeled image)
labeled_image = np.array([
	[0, 0, 1, 1, 0, 0, 0, 2, 2, 0],
	[0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
	[1, 1, 1, 1, 1, 0, 0, 2, 2, 0],
	[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
	[0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
	[0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# Compute density map using weighted KDE
density_map = weighted_kde_density_map(labeled_image)

density_map = density_map / density_map.max()

# Visualize density map
plt.imshow(density_map, vmin = 0, vmax = 1, cmap='viridis')
plt.colorbar()
plt.title('Density Map')
plt.xticks([])
plt.yticks([])
plt.show()

# Visualize density map
plt.imshow(labeled_image, vmin = 0, cmap='viridis')
plt.colorbar()
plt.title('Labelled Image')
plt.xticks([])
plt.yticks([])
plt.show()