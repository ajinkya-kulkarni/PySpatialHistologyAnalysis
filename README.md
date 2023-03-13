[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Demonstrating PySpatialAnalysis using [StarDist](https://github.com/stardist/stardist)

## App Overview

![Streamlit App Screenshot](https://github.com/ajinkya-kulkarni/PySpatialAnalysis/blob/main/screenshot.png)

This is a web application for analyzing H&E images using the PySpatialAnalysis package and [StarDist](https://github.com/stardist/stardist). The app allows the user to upload an H&E image, and it performs object (nuclei) detection on the image using the StarDist2D model. The detected objects are then highlighted and displayed alongside the original image.

The app is built using the Streamlit framework and requires the following dependencies:
- streamlit
- PIL
- numpy
- BytesIO
- modules (custom module for the app)

To run the app, navigate to the directory containing the code and run the following command:
`streamlit run PySpatialAnalysis_StreamlitApp.py`

Upon running the command, the app will open in your browser at `localhost:8501`.

To use the app, simply upload an H&E image using the file upload widget and click the "Analyze" button. The app will then perform object detection on the image using the StarDist2D model and display the results alongside the original image.

If an error occurs during image analysis, an error message will be displayed.

Note that the app works best for images smaller than 1000x1000 pixels.

Enjoy using PySpatialAnalysisWSI and StarDist web app!

#### References:

1. All images taken from [Link 1](https://twitter.com/JMGardnerMD) and [Link 2](https://twitter.com/kiko4docs)