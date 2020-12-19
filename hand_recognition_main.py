import os
import numpy as np
import cv2
from enum import Enum

from data_preprocessing import DataPreprocessing
from feature_extraction import FeatureExtraction

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
   
    # Print New Line on Complete
    if iteration == total: 
        print()

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

    def loadImageCv(self,id):
        if self.imagesList_dir:
            path = self.dataset_dir+"//"+self.imagesList_dir[id]
            if os.path.exists(path):
                return cv2.imread(path)    

    def __init__(self, datasetFolder):
               
        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, datasetFolder) 
        self.getImagesToLoad()


def main():
    #Load image filenames from Data folder
    dLoader_obj = DataLoader("Data")

    #Describe images - filename, class, author
    dLoader_obj.describeLoadedData()

    #print first 10    
    print(*dLoader_obj.dataset_array[0:10], sep="\n")
    
    #load images as opencv Mat 
    imgNum = 1
    dLoader_obj.loadImagesCv(imgNum)

    #load images one by one, resize and save
    #set path to save processed images 
    outPut_dir = os.path.join(dLoader_obj.project_dir,"Processed") 

    printProgressBar(0, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj.imagesList_dir)):
        img = dLoader_obj.loadImageCv(i)
        resized_img = DataPreprocessing.resizeImage(img,120)
        DataPreprocessing.save_image(resized_img,dLoader_obj.dataset_array[i],"ResizedImages",outPut_dir,95)
        printProgressBar(i + 1, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    
    # dPrep_obj = DataPreprocessing(dLoader_obj.imagesList_cv, dLoader_obj.dataset_array[0:imgNum], outPut_dir)

    # #resize images (scale_down) and save to file
    # shortEdgeLength = 120
    # resizedImages = dPrep_obj.resizeImages(shortEdgeLength)

    # #save resized images to file for further processing
    # dPrep_obj.save_processed_images(resizedImages, "ResizedImages",80)

    # #skin detection for segmentation
    # COLORSPACE = Enum('Colorspace', 'HSV YUV YCBCR') 
    # _, tresh = dPrep_obj.skinDetection(COLORSPACE.HSV,resizedImages[0])

    # cv2.imshow("treshHsv", tresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
