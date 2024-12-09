**Table of Contents**
1.	Project Title
2.	Project Description
3.	Team Members
4.	Features
5.	Software Used
6.	Installation and Setup
•	Prerequisites
•	Steps
7.	Input Image Requirements
8.	Operating System
9.	How to Use
•	Load an Image
•	Submit for Processing
•	View Results
•	Clear Output
10.	File Descriptions
•	Main Files
•	Supporting Files
11.	Expected Output
12.	References and Resources

**Project Title**
Morphological Matching for Similar Image Detection

**Project Description**

This project is designed to detect and retrieve similar images from a dataset using a pretrained VGG16 model for feature extraction. A user-friendly GUI, built with **Tkinter**, allows users to:  
-	Upload an image for analysis.
-	Process the image using the trained model.
-	View the top 10 similar images with their labels and similarity scores.

**Team Members**
Yaswanth Bharath Soma (Z23744408)
Naga Keerthi Thota (Z23789716)
Rishika Reddy Aleti (Z23741614)
Niharika Janardhan Konduru (Z23812515)

**Features**
-	Extracts image embeddings using a pretrained VGG16 model.
-	Calculates the cosine similarity between the input image and the dataset images.
-	Displays the top 10 similar images with labels and similarity scores.
-	Provides an interactive GUI for user-friendly image upload and visualization.
-	Includes error handling with warnings and user-friendly messages.


**Software Used**
Python 3.7+

TensorFlow/Keras: For deep learning-based image feature extraction.
OpenCV: For image preprocessing and manipulation.
Pillow (PIL): For image handling in the GUI.
Tkinter: For building the graphical user interface.
NumPy: For numerical computations.
Pandas: For handling and analyzing the dataset.
SciPy: For calculating similarity metrics.

**Installation and Setup**
Prerequisites
1.	Install Python 3.10 on your system.
2.	Install pip (Python's package manager) if not already installed..

Steps
1.Clone the repository to your local machine:
Code
-	git clone https://github.com/niharikajk2002/Morphological-Matching
-	cd Morphological-Matching

2.Install the required Python packages:
Code
-	pip install -r requirements.txt

3.	Place the following required files in the project directory:
-	vgg16_feature_extractor.h5: Pretrained VGG16 model for feature extraction.
-	features_matrix.npy: Precomputed feature embeddings for the dataset.
-	dataset_df.csv: Metadata of the dataset containing file paths and labels.

4.	Run the program:
Code
-	python finalgui.py

**Input Image Requirements**
-	Accepted Formats: .jpg, .jpeg, .png, .bmp, .tiff
-	Recommended Dimensions: Images will be resized to 224x224 pixels for processing.

**Operating System**
The program has been tested on the following platforms:
-	Windows 10/11
-	Linux (Ubuntu 20.04 or later)

**How to Use**
1.	Load an Image:
Click the "Load Image" button to upload an image file from your computer.
2.	Submit for Processing:
Click the "Submit" button to compute and retrieve the top 10 similar images from the dataset. A progress window will appear during the computation.
3.	View Results:
The application displays the top 10 similar images with their paths, labels, and similarity scores in the output area.
Thumbnails of the similar images are displayed for quick visualization.
4.	Clear Output:
Click the "Clear Output" button to reset the interface for a new query.

**File Descriptions**
Main Files
finalgui.py: Main script to run the application. Implements the GUI and similarity detection functionality.
vgg16_feature_extractor.h5: Pretrained VGG16 model for extracting image embeddings.
features_matrix.npy: Numpy file containing precomputed feature embeddings of the dataset.
dataset_df.csv: CSV file containing image metadata (paths and labels).

Supporting Files
requirements.txt: Contains a list of Python packages required for the project.

**Expected Output**
1.	The application will display the top 10 similar images from the dataset along with:
-	Image Paths
-	Labels
-	Cosine Similarity Scores
2.	Thumbnails of the top 10 similar images will be displayed in the GUI.

**References and Resources**
1.	VGG16 Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
2.	Tkinter Documentation: https://docs.python.org/3/library/tkinter.html
3.	OpenCV Documentation: https://opencv.org/
4.	SciPy Documentation: https://docs.scipy.org/doc/scipy/

