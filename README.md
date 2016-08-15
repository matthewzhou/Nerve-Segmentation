Overview
============
A CNN that reads in ultrasound images and manually labeled masks delineating nerve regions and reconstructs nerve masks for test ultraound images.

Dependencies
============
* Python 2.7+ (pip install python2.7)
* opencv/cv2 (install through Homebrew -- Mac installation follow instructions here: https://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/)
* keras (pip install keras)
* Theano (pip install Theano)
* sklearn (pip install sklearn)
* Numpy (pip install numpy)


Basic Usage
===========
Download data from https://www.kaggle.com/c/ultrasound-nerve-segmentation/data (File Size ~ 1 GB)
Samples of test and train data present in folder

root directory should look like:

../P5_Submission_Folder
-raw
 |
 ---- trainsample
 |    |
 |    ---- 1_1.tif
 |    |
 |    ---- …
 |
 ---- testsample
      |
      ---- 1.tif
      |
      ---- …

Change file paths in capstone.py that are marked by comments
Run python capstone.py from terminal 