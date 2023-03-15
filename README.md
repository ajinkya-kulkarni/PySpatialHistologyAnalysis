[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyspatialhistologyinformation.streamlit.app/)
[![DOI](https://zenodo.org/badge/612892393.svg)](https://zenodo.org/badge/latestdoi/612892393)

## Demonstrating PySpatialHistologyAnalysis using [StarDist](https://github.com/stardist/stardist)

#### App Overview

This is primarily a [web application](https://pyspatialhistologyinformation.streamlit.app/) for analyzing H&E images using the PySpatialHistologyAnalysis package, which utilizes the [StarDist](https://github.com/stardist/stardist) and the [PySAL](https://pysal.org/) packages under it's hood. The app allows the user to upload an H&E image, and it performs object (nuclei) detection on the image using the StarDist2D model. The detected objects are then highlighted and displayed alongside the original image. Spatial analysis is then performed on the detected nuclei, and finally a spreadsheet of all the results is prepared.

![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot1.jpg)

![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot2.jpg)

#### Dependencies

The app is built using the Streamlit framework and requires the following dependencies:
- stardist
- streamlit
- Pillow
- numpy
- BytesIO
- tensorflow-cpu
- scipy
- pandas
- matplotlib
- scikit-image
- scikit-learn
- modules (custom module for the app)

#### Using the app

To run the app, navigate to the directory containing the code and run the following command:
`streamlit run PySpatialAnalysis_StreamlitApp.py`

Upon running the command, the app will open in your browser at `localhost:8501`.

To use the app, simply upload an H&E image using the file upload widget and click the "Analyze" button. The app will then perform object detection on the image using the StarDist2D model and display the results alongside the original image.
If an error occurs during image analysis, an error message will be displayed.
Note that the app works best for images smaller than 1000x1000 pixels.


#### References:

1. All images taken from [Link 1](https://twitter.com/JMGardnerMD) and [Link 2](https://twitter.com/kiko4docs)
