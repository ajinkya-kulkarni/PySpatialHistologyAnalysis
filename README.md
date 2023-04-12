[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyspatialhistologyinformation.streamlit.app/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7810805.svg)](https://doi.org/10.5281/zenodo.7810805)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/ajinkya-kulkarni/PySpatialHistologyAnalysis)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/ajinkya-kulkarni/PySpatialHistologyAnalysis)
![GitHub all releases](https://img.shields.io/github/downloads/ajinkya-kulkarni/PySpatialHistologyAnalysis/total)
![GitHub language count](https://img.shields.io/github/languages/count/ajinkya-kulkarni/PySpatialHistologyAnalysis)

## Demonstrating PySpatialHistologyAnalysis using [StarDist](https://github.com/stardist/stardist)

#### App Overview

This is primarily a [web application](https://pyspatialhistologyinformation.streamlit.app/) for analyzing H&E images using the PySpatialHistologyAnalysis package, which utilizes the [StarDist](https://github.com/stardist/stardist) packages under it's hood. 
The application allows the user to upload an H&E image. It first stain normalizes the image and then performs object (nuclei) detection on the image using the [StarDist2D](https://github.com/stardist/stardist) model. 
The detected objects are then highlighted and displayed alongside the original image. 
Spatial analysis is then performed on the detected nuclei, and finally a spreadsheet of all the results is prepared.

![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot1.png)
![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot2.png)
![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot3.png)
![App Screenshots](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis/blob/main/screenshot4.png)

#### Dependencies

The app is built using the Streamlit framework and requires the dependencies as mentioned in the `requirements.txt` file.

#### Using the app

To use the app, simply upload an H&E image using the file upload widget and click the "Analyze" button. The app will then perform object detection on the image using the [StarDist2D](https://github.com/stardist/stardist) model and display the results alongside the original image.
If an error occurs during image analysis, an error message will be displayed.
Note that the app works best for images smaller than 1000x1000 pixels.


#### References:

1. All images taken from [Link 1](https://twitter.com/JMGardnerMD) and [Link 2](https://twitter.com/kiko4docs)
