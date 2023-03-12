import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

import numpy as np
import glob
import os

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##################################################################################

from modules import *

##################################################################################

os.system('cls || clear')

# Glob all the JPEG files in the directory
jpeg_files = glob.glob("*.jpeg")

# Loop over the JPEG files

for jpeg_file in jpeg_files:
    
    print('Current file: ' + jpeg_file)
    
    rgb_image, labels, more_info, rendered_labels = make_analysis(jpeg_file)
    
    ############################################################

    modified_labels = np.where(labels > 0, 1, labels)
    
    ############################################################

    fig = plt.figure(figsize = (5, 5), constrained_layout = True)

    ax_array = fig.subplots(1, 2, squeeze = False)

    im = ax_array[0, 0].imshow(rgb_image)

    im = ax_array[0, 1].imshow(modified_labels, cmap = 'cividis', vmin = 0, vmax = 1)

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

    plt.savefig(result_filename, dpi = 200)
    
    plt.close()
    
    print()
    
    ############################################################
    
print()
print('All done!')
print()

##################################################################################