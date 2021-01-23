import cv2
import numpy as np
import math 
import string
import os
import joblib

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colorbar

class FeatureExtraction:
    """Class provides tools to extract features from images."""
    pass

    @staticmethod
    def applyPCA(trainFeatures, testFeatures, varianceRetained=0.95):
        scaler = StandardScaler()
        scaler.fit(trainFeatures) # fit only for training data

        trainFeatures = scaler.transform(trainFeatures)
        testFeatures = scaler.transform(testFeatures)

        pca = PCA(varianceRetained)
        pca.fit(trainFeatures) # fit only for training data
        print(f"PCA maps for: {pca.n_components_} components and retains: {varianceRetained} variance\n")  
        filename = 'pca.sav'
        joblib.dump(pca, filename)

        trainFeatures = pca.transform(trainFeatures)
        testFeatures = pca.transform(testFeatures)

        return trainFeatures, testFeatures 

    @staticmethod
    def applyPCA_signle(features, varianceRetained=0.95):
        scaler = StandardScaler()
        scaler.fit(features) # fit only for training data

        modelFeatures = scaler.transform(features)

        pca = PCA(varianceRetained)
        pca.fit(modelFeatures)
        
        pcaFeatures = pca.transform(modelFeatures)

        return pcaFeatures 

    @staticmethod
    def project2D_PCA(allFeatures, labels):

        dir = os.getcwd()
        plot_dir = os.path.join(dir, "Plots")

        pca = PCA(2)
        projected = pca.fit_transform(allFeatures)

        lst = list(set(labels)) # get unique values
        classNames = sorted(lst)

        labelsNum = [ord(c)-65 for c in labels]

        plt.scatter(projected[:, 0], projected[:, 1],
            c=labelsNum, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', len(classNames)))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        myTicks = sorted(list(set(labelsNum)))
        cbar = plt.colorbar(orientation = 'vertical')   
        cbar.set_ticks(myTicks)   
        cbar.set_ticklabels(classNames)
        plt.savefig(plot_dir + "//pca2D_hog.png", dpi = 300)

    @staticmethod
    def getCentroid(contour):
        
        M = cv2.moments(contour)   
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx,cy)
   

    @staticmethod
    def getAdamFeatures(cnt, binary_image):
        features_dict = {}
        M = cv2.moments(cnt)
        
        # Centroid
        c_x = int(M['m10'] / M['m00'])
        c_y = int(M['m01'] / M['m00'])
        features_dict['centroid_x'] = c_x
        features_dict['centroid_y'] = c_y
        
        hand_area = cv2.contourArea(cnt)
        features_dict['hand_area'] = hand_area
        hand_perimeter = cv2.arcLength(cnt, True)
        features_dict['hand_perimeter'] = hand_perimeter
        hand_area2per = hand_area/hand_perimeter
        features_dict['hand_area2per'] = hand_area2per

        # Straight bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w * h
        features_dict['rect_area'] = rect_area

        # Minimum enclosing circle
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        circle_area = math.pi * radius * radius
        features_dict['circle_area'] = circle_area

        # Straight line - orientation
        rows, cols = binary_image.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x) * vy / vx) + y)
        line_slope = (lefty - righty) / (0 - (cols-1))
        features_dict['line_slope'] = line_slope

        return features_dict

    @staticmethod
    def getHog(image,_orientations=9, _pixels_per_cell=(8, 8), _cells_per_block=(2, 2), _visualize=True, _multichannel=True):
        fd, img = hog(image, orientations=_orientations, pixels_per_cell=_pixels_per_cell, cells_per_block=_cells_per_block, visualize=_visualize, multichannel=_multichannel)
        return fd, img

    @staticmethod
    def get_hog_from_imageset(image_set, images_description):
        """
        @param: image_set - images for hog extraction
        @param: images_description - labels for image_set. 
        
        @return: hog_data - return tuple with (hog, image) structure
                 fd_set - list with hogs for all image_set
                 img_description_set - same as images_description
        """
        fd_set =[]
        img_description_set = []
        hog_data = []
        for image in range(0, len(images_description)):

            fd, _ = hog(image_set[image], orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True, multichannel=True)
            fd_set.append(fd)
            img_description_set.append(images_description[image][0])
            hog_data.append((fd, images_description[image][0]))
        return hog_data, fd_set, img_description_set

    @staticmethod
    def featureDict2x(feat_dict):
        x = []
        for feat in list(feat_dict.values()):
            x.append(feat)
        return x

    def __init__(self):
        pass
