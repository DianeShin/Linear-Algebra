# Import necessary packages
import os
import sys
import cv2
import random
import time
import math
import pickle 
import numpy as np
import urllib.request
import os.path
from helper import *
from asg import *

# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
        # Start with the mean image
        output = averageFace
        
        # Add the eigen faces with the weights
        for i in range(0, NUM_EIGEN_FACES):
                '''
                OpenCV does not allow slider values to be negative. 
                So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
                ''' 
                sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
                weight = sliderValues[i] - MAX_SLIDER_VALUE/2
                output = np.add(output, eigenFaces[i] * weight)

        # Display Result at 2x size
        output = cv2.resize(output, (0,0), fx=2, fy=2)
        cv2.imshow("Result", output)

def resetSliderValues(*args):
        for i in range(0, NUM_EIGEN_FACES):
                cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));    
        createNewFace()


def showOrigAndBackProjected(origData, backProjData, shape):
    #nph = np.hstack((origData.reshape(shape), backProjData.reshape(shape)))
    nphc= np.concatenate((origData.reshape(sz), backProjData.reshape(shape)), axis=1)
    cv2.imshow("Result",nphc)


        
if __name__ == '__main__':

        # Number of EigenFaces 
        NUM_EIGEN_FACES = 10

        # Maximum weight
        MAX_SLIDER_VALUE = 255


        # Read images from a file created via the pickle module
        #
        # If it does not exist locally, we fetch it from the IMADA web server (once)
        #
        if os.path.isfile("images.p"):
                print("Images pickle file exists. Using local copy.")
                images = pickle.load( open( "images.p", "rb" ) )
        else:
                print('''
Before running this program successfully you need to download a file containing all images. You have two options:

1.) (recommended) Please download the file 'https://imada.sdu.dk/~daniel/DM561-DM562-images/images.p' and put it in the current directory, i.e.,  next to 'asg6.py' (do not change the filename). When the file exists this message will disappear. 

2.) Uncomment the two lines in the code 'EigenFace.py' just below this print message and remove the 'sys.exit()'. This will automatically download the necessary file for you.


The downloaded file 'images.p' has a size of 77423721 bytes and the md5 hash is 0184581eb1a8d94f20bb1c44a5e3ec45. For the final grading we might use another set of images. 
                ''')
                 
                images = pickle.load(urllib.request.urlopen("https://imada.sdu.dk/~daniel/DM561-DM562-images/images.p"))
                pickle.dump( images, open( "images.p", "wb" ) )
                

        # Alternatively you can read image files directly, this should not
        # be necessary, though
        #
        # Directory containing images
        # dirName = "images/"
        
        # Read images
        # images = readImages(dirName)
        # see https://imada.sdu.dk/~daniel/DM561-DM562-images/pickle-creation.txt on how the file images.d was created


        # 
        # Size of images
        sz = images[0].shape

        # Create data matrix for PCA.
        data = createDataMatrix(images)

        # Compute the eigenvectors from the stack of images created
        print("Calculating PCA ", end="...")
        mean, eigenVectors = cv2.PCACompute(data[20:], mean=None, maxComponents=NUM_EIGEN_FACES)

        # If you have a new version of opencv installed, you can also use 
        # mean, eigenVectors, eigenValues = cv2.PCACompute2(data[20:], mean=None, maxComponents=NUM_EIGEN_FACES)
        print ("DONE")

        averageFace = mean.reshape(sz)

        eigenFaces = []; 

        for eigenVector in eigenVectors:
                eigenFace = eigenVector.reshape(sz)
                eigenFaces.append(eigenFace)

        # Create window for displaying Mean Face
        cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
        
        # Display result at 2x size
        output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
        cv2.imshow("Result", output)

        # Create Window for trackbars
        cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

        sliderValues = []
        
        # Create Trackbars
        for i in range(0, NUM_EIGEN_FACES):
                sliderValues.append(MAX_SLIDER_VALUE/2)
                cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
        
        # You can reset the sliders by clicking on the mean image.
        cv2.setMouseCallback("Result", resetSliderValues);
        
        print('''Usage:
        Change the weights using the sliders
        Click on the result window to reset sliders
        Hit ESC to terminate program.''')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # The following code will only work, if you have a functional
        # implementation of asg6 methods.
        for i in range(0,20):
                orig =  data[i,:]
                back = projectAndBackProject(orig, mean, eigenVectors)
                showOrigAndBackProjected(orig, back, sz)
                print("Distance:",distanceImages(orig,back))
                cv2.waitKey(0)

        l=indexDistancePairsSortedByDistance(data, range(0,20), mean, eigenVectors)
        print("The picture furthest away from the eigenface space has id ",l[-1])
        
