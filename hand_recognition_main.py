import os
import numpy as np
import cv2
from enum import Enum

from data_preprocessing import DataPreprocessing
from feature_extraction import FeatureExtraction


class DataLoader:
    """Class to load data from dataset."""
    
    project_dir = ""
    dataset_dir = ""
    imagesList_dir = []
    imagesList_cv = []
    dataset_array = []

    def getImagesToLoad(self):

        if os.path.isdir(self.dataset_dir):            
            self.imagesList_dir = os.listdir(self.dataset_dir) 

            if self.imagesList_dir:
                print ("Found: {0} images in folder.".format(len(self.imagesList_dir)))
        else:
            print("Provided directory does not exist!")

    def describeLoadedData(self):
        if self.imagesList_dir:

            for elem in self.imagesList_dir:
                filename = elem.split('.')[0]
                author = ""

                if "ad" in filename:
                    author = "Adam"
                elif "(T)" in filename:
                    author = "Tomek"
                elif "OD" in filename:
                    author = "Oleksandr"
                else:
                    author = "Oleksandr_2"    
                
                #filename(with file extension), class, within class id, author of photo
                self.dataset_array.append([elem, elem[0], elem[2], author])
        else:
            print("Provided list is empty")

    def loadImagesCv(self, im_num=0):
      
        if self.imagesList_dir and not self.imagesList_cv:
        
            loadedImages = 0
            for elem in self.imagesList_dir:
                img = cv2.imread(self.dataset_dir+"//"+elem)
                self.imagesList_cv.append(img)
                loadedImages += 1
                if loadedImages >= im_num and im_num !=0:
                    break
            print ("Loaded: {0} images.".format(len(self.imagesList_cv)))
        else:
            print("List of images to load is empty or images are already loaded")

    def __init__(self, datasetFolder):
               
        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, datasetFolder) 
        self.getImagesToLoad()


def main():
    #Load image filenames from Data folder
    dLoader_obj = DataLoader("Data")

    #Describe images - filename, class, author
    dLoader_obj.describeLoadedData()

    #print first
    print(*dLoader_obj.dataset_array[0:10], sep="\n")

    #load images as opencv Mat 
    imgNum = 1
    dLoader_obj.loadImagesCv(imgNum)
        
    #setup preprocessing
    #Defalut path to save processed images
    outPut_dir = os.path.join(dLoader_obj.project_dir,"Processed")
    dPrep_obj = DataPreprocessing(dLoader_obj.imagesList_cv, dLoader_obj.dataset_array[0:imgNum], outPut_dir)

    #resize images (scale_down) and save to file
    shortEdgeLength = 120
    resizedImages = dPrep_obj.resizeImages(shortEdgeLength)

    #save resized images to file for further processing
    dPrep_obj.save_processed(resizedImages, "ResizedImages",80)

    #skin detection for segmentation
    COLORSPACE = Enum('Colorspace', 'HSV YUV YCBCR') 
    _, tresh = dPrep_obj.skinDetection(COLORSPACE.HSV,resizedImages[0])

    cv2.imshow("treshHsv", tresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()