import os
os.system('cls || clear')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
import numpy as np
import glob
import cv2

from stardist import relabel_image_stardist

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##################################################################################

from modules import *

##################################################################################

# Glob all the JPEG files in the directory
jpeg_files = sorted(glob.glob("Test_Files/*.jpeg"))

# Loop over the JPEG files

for jpeg_file in jpeg_files:
    
    print('Current file: ' + jpeg_file)
    
    rgb_image, labels, more_info, rendered_labels = make_analysis(jpeg_file)
    
    ############################################################

    relabelled_image = relabel_image_stardist(labels, n_rays = 128)

    modified_labels = np.where(relabelled_image > 0, 1, relabelled_image)
    
    ############################################################

    # new_size = (800, 800)  # example size

    # resized_modified_labels = cv2.resize(modified_labels, new_size, interpolation=cv2.INTER_NEAREST)

    # resized_rgb_image = cv2.resize(rgb_image, new_size)

############################################################

    fig = plt.figure(figsize = (8, 4), constrained_layout = True)

    ax_array = fig.subplots(1, 2, squeeze = False)

    im = ax_array[0, 0].imshow(rgb_image)

    im = ax_array[0, 1].imshow(modified_labels, cmap = 'binary', vmin = 0, vmax = 1)

    ax_array[0, 0].set_title('Original Image')
    ax_array[0, 1].set_title('Segmented Image, N= ' + str(len(more_info['points'])))

    ax_array[0, 0].set_xticks([])
    ax_array[0, 0].set_yticks([])

    ax_array[0, 1].set_xticks([])
    ax_array[0, 1].set_yticks([])
    
    ############################################################
    
    result_filename = jpeg_file[:-5] + '_result.png'
        
    ############################################################
    
    if os.path.exists(result_filename):
        os.remove(result_filename)
        
    ############################################################

    plt.savefig(result_filename, dpi = 300, bbox_inches='tight')
    
    plt.close()
    
    print()
    
    ############################################################

# convert_png_to_pdf("output.pdf")

############################################################

print()
print('All done!')
print()

##################################################################################