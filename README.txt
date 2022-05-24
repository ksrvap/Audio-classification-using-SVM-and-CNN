About the script:
-----------------
-The language used to implement the Audio Classification is "python".
-There are two scripts uploaded that contain the code for the classifiers.
-One of them is a notebook document and the other is a python script and the names are "Audio_Classification.ipynb" and "Audio_Classification.py" respectively.
-The code in the notebook file can be seen as sections for better understanding.
-Both of them are executable under different environments and the instructions for execution are given below.

Instructions for execution:
---------------------------
Requirements:
1. Python 3.7 or above installed (for .py file)
2. Jupyter Notebook or Google Colab (for .ipynb file)
3. Or any other IDE that supports python execution.

Steps to execute:
-----------------
1. First of all, install the required packages for this project. Some of the packages are:
  - librosa
  - keras
  - tensorflow
  - ffmpeg

2. Load the file into the environment, either .py file or .ipynb file and import the required libraries.

3. Make sure to have a folder which contains 
  - The folders train and test which have training and tesing audio files.
  - The csv files train.csv and test_idx.csv.

4. Importing and converting audio files
  - In the "Extract Training and Testing Audio files into Numpy Arrays" section,
  - Change the path for the project folder mentioned in above step.
  - Copy the path of the folder in your system and store it as the value of a string variable (original_path).
  - This section has calls for two methods get_training() and get_testing() which converts the audio files into numpy arrays.
  - The resulting numpy arrays are stored in train_extracted and test_extracted folders inside the main folder.  

5. For implementing different data representations
  - We have decided to represent the data in 3 different ways by using PCA, MFCC, and Spectrograms.
  - We have defined different functions to extract the required features for each of these different data representations.
   
6. For implementing different classifiers
  - The three classifiers implemented in this project are
   i). Support Vector Machines(SVM) using Principal Component Analysis(PCA)
   ii). Random Forest using Mel-Frequency Cepstral Coefficients(MFCC)
   iii). 2-Dimensional Convolutional Neural Network(CNN) using Spectrograms.
  - There are functions defined to implement each of these classifiers.

7. For reporting the accuracy of the model
  - We have defined a function to calculate and display the confusion matrix of the classifiers.
  - There is also a function to perform 5-fold cross validation and report the accuracy at 99% confidence interval. 

8. For training and testing the classifier
  - There are three different sections for the three classifiers.
  - Executing each of these sections will result in the generation of a csv file which has the predictions for testing data.
  - The names of the predicted csv files are predicted_svm, predicted_rf, and predicted_cnn.
  - These names can be changed accordingly in the respective sections.

9. After following all the above steps, run all the cells(for .ipynb file) or run the entire code(for .py file).

10. The three output files w.r.t three classifiers are generated in the main folder.