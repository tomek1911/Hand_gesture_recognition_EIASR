import os
import numpy as np
import cv2
import csv
import pandas as pd

from enum import Enum
from data_preprocessing import DataPreprocessing as dp
from feature_extraction import FeatureExtraction as fe
from data_classification import DataClassification as dc

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

    def describeLoadedDataPNG(self):

        self.dataset_array = []
        if self.imagesList_dir:

            for elem in self.imagesList_dir:
                filename = elem.split('.')[0]          
                extension = elem.split('.')[1]      
                split = filename.split('_')

                label = split[0]
                idInClass = split[1]
                authorCode = split[2]               
          
                self.dataset_array.append([label, idInClass, authorCode, filename, extension])
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

    def loadImageCvGray(self,id):
        if self.imagesList_dir:
            path = self.dataset_dir+"//"+self.imagesList_dir[id]
            if os.path.exists(path):
                return cv2.imread(path,cv2.IMREAD_GRAYSCALE)   

    def dictToCsv(self, dict_data, dict_keys, folder, filename):

        myfile = filename + ".csv"
        path = os.path.join(self.project_dir, folder, myfile) 
        try:
            with open(path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dict_keys)
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")


    def __init__(self, datasetFolder):

        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, datasetFolder) 
        self.getImagesToLoad()

    @staticmethod
    def manualTresholdTester(image):
    
        title_window = "Test slider"
        slider_maxH = 255
        slider_maxS = 255
        slider_maxV = 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        tNameCh1_low = 'Ch1_low'
        tNameCh1_high = 'Ch1_high'
        tNameCh2_low = 'Ch2_low'
        tNameCh2_high = 'Ch2_high'
        tNameCh3_low = 'Ch3_low'
        tNameCh3_high = 'Ch3_high'
        
        def on_trackbar(val):
            ch1_low_slider=cv2.getTrackbarPos(tNameCh1_low, title_window)
            ch1_high_slider=cv2.getTrackbarPos(tNameCh1_high, title_window)
            ch2_low_slider=cv2.getTrackbarPos(tNameCh2_low, title_window)
            ch2_high_slider=cv2.getTrackbarPos(tNameCh2_high, title_window)
            ch3_low_slider=cv2.getTrackbarPos(tNameCh3_low, title_window)
            ch3_high_slider=cv2.getTrackbarPos(tNameCh3_high, title_window)

            lowerBound = (ch1_low_slider,  ch2_low_slider, ch3_low_slider)
            upperBound = (ch1_high_slider,  ch2_high_slider, ch3_high_slider)
            img_tresholded = cv2.inRange(image, lowerBound,  upperBound)
            cv2.imshow(title_window,img_tresholded)

        cv2.namedWindow(title_window,cv2.WINDOW_NORMAL & cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(title_window,image)

        cv2.createTrackbar(tNameCh1_low, title_window , 0, slider_maxH, on_trackbar)
        cv2.createTrackbar(tNameCh1_high, title_window , 0, slider_maxH, on_trackbar)
        cv2.createTrackbar(tNameCh2_low, title_window , 0, slider_maxS, on_trackbar)
        cv2.createTrackbar(tNameCh2_high, title_window , 0, slider_maxS, on_trackbar)
        cv2.createTrackbar(tNameCh3_low, title_window , 0, slider_maxV, on_trackbar)
        cv2.createTrackbar(tNameCh3_high, title_window , 0, slider_maxV, on_trackbar)

        cv2.setTrackbarPos(tNameCh1_low, title_window,50)
        cv2.setTrackbarPos(tNameCh1_high, title_window,255)
        cv2.setTrackbarPos(tNameCh2_low, title_window,140)
        cv2.setTrackbarPos(tNameCh2_high, title_window,180)
        cv2.setTrackbarPos(tNameCh3_low, title_window,60)
        cv2.setTrackbarPos(tNameCh3_high, title_window,130)

        cv2.waitKey(0)
        cv2.destroyAllWindows()     


def main():
    #Load image filenames from Data folder
    dLoader_obj = DataLoader("Data")

    #Describe images - filename, class, author
    dLoader_obj.describeLoadedData()

    #print first 10    
    print(*dLoader_obj.dataset_array[0:20], sep="\n")
    
    #load images as opencv Mat 
    imgNum = 1
    dLoader_obj.loadImagesCv(imgNum)

    #load images one by one, resize and save
    #set path to save processed images 
    outPut_dir = os.path.join(dLoader_obj.project_dir,"Processed")    

    #Test tresholds for data_processing
    
    # image = dLoader_obj.imagesList_cv[8]
    # DataLoader.manualTresholdTester(image)

    ##########################################################################################################################
    #RESIZE


    # printProgressBar(0, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    # for i in range(0,len(dLoader_obj.imagesList_dir)):
    #     img = dLoader_obj.loadImageCv(i)
    #     resized_img = dp.resizeImage(img,120)
    #     dp.save_image(resized_img,dLoader_obj.dataset_array[i],"ResizedImages",outPut_dir,95,"png")
    #     printProgressBar(i + 1, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    ##########################################################################################################################
    #THRESHOLD                

    dLoader_obj_resized = DataLoader("Processed/ResizedImages")   
    dLoader_obj_resized.describeLoadedDataPNG()
    dLoader_obj_resized.loadImagesCv()

    printProgressBar(0, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)
    # tresholdedImagesList = []

    for i in range(0,len(dLoader_obj_resized.imagesList_dir)):
        img = dLoader_obj_resized.loadImageCv(i)
        tresholdedImage = dp.tresholdImageYCBCR(img)
        dp.save_image3(tresholdedImage,dLoader_obj_resized.dataset_array[i],"TresholdedImages",outPut_dir,95,"png")
        # tresholdedImagesList.append(tresholdedImage)
        printProgressBar(i + 1, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    #########################################################################################################################
    #RESULTS PREVIEW
    
    dLoader_obj_tresh = DataLoader("Processed/TresholdedImages")   
    dLoader_obj_tresh.describeLoadedDataPNG()
    dLoader_obj_tresh.loadImagesCv() #wczytaj wszystkie zdjecia z folderu
    
    tresholdedImagesList = dLoader_obj_tresh.imagesList_cv    
     
    # fullimg = np.zeros((0,2520),np.uint8)
    # fullimg_morph = np.zeros((0,2520),np.uint8)

    # for i in range (0,9):   
    #     img = np.zeros((160,0),np.uint8)
    #     img_morph = np.zeros((160,0),np.uint8)
    #     for j in range(0,21):   
    #         ind = j + i*21             
    #         img = np.concatenate((img, tresholdedImagesList[ind]), axis = 1)
            
    #         img_filtered = dp.morphologicFiltering(tresholdedImagesList[ind],(5,5))

    #         img_morph = np.concatenate((img_morph, img_filtered), axis = 1)
        
    #     fullimg = np.concatenate((img, fullimg), axis = 0)
    #     fullimg_morph = np.concatenate((img_morph, fullimg_morph), axis = 0)
    
    # dp.save_image2(fullimg,"zestawienie_progowanie3","",outPut_dir,95)
    # dp.save_image2(fullimg_morph,"zestawienie_otwarcie_zamkniecie3","",outPut_dir,95)

    #########################################################################################################################
    #MORPHOLOGIC FILTER

    printProgressBar(0, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_tresh.imagesList_dir)):
        img = dLoader_obj_tresh.loadImageCv(i)
        morph_img = dp.morphologicFiltering(img, (5,5))
        dp.save_image3(morph_img,dLoader_obj_tresh.dataset_array[i],"MorphFilter",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_tresh.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    #########################################################################################################################
    #CONTOURS

    dLoader_obj_binary = DataLoader("Processed/MorphFilter")   
    dLoader_obj_binary.describeLoadedDataPNG()
    dLoader_obj_binary.loadImagesCv() #wczytaj wszystkie zdjecia z folderu
    featuresList = []       
  
    # printProgressBar(0, len(dLoader_obj_binary.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    # for i in range(0,len(dLoader_obj_binary.imagesList_dir)):
    #     img = dLoader_obj_binary.loadImageCvGray(i)
    #     featuresList.append(fe.getAdamFeatures(img)) #TODO - opis wewnatrz funkcji
    #     printProgressBar(i + 1, len(dLoader_obj_binary.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    contoursList=[]

    printProgressBar(0, len(dLoader_obj_binary.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_binary.imagesList_dir)):
        img = dLoader_obj_binary.loadImageCvGray(i)
        contour = dp.filterContours(img)
        contoursList.append(contour)
        output = np.zeros((160,120,3), np.uint8)
        cv2.fillPoly(output, pts =[contour], color=(255,255,255))
        #cv2.drawContours(output, contour, -1, (0, 0, 255), 2) 
        dp.save_image3(output,dLoader_obj_binary.dataset_array[i],"Contours",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_binary.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)   

    #########################################################################################################################
    #CONTOURS

    dLoader_obj_cont = DataLoader("Processed/Contours")   
    dLoader_obj_cont.describeLoadedDataPNG()
    dLoader_obj_cont.loadImagesCv() #wczytaj wszystkie zdjecia z folderu
    featuresList = []    

    printProgressBar(0, len(dLoader_obj_cont.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_cont.imagesList_dir)):
        img = dLoader_obj_cont.loadImageCvGray(i)
        cnt = contoursList[i]
        dict_adam = fe.getAdamFeatures(cnt,img) 
        dict_adam['label'] = dLoader_obj_cont.dataset_array[i][0]
        featuresList.append(dict_adam)
        printProgressBar(i + 1, len(dLoader_obj_cont.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)   

    df_features = pd.DataFrame(featuresList)
    project_path = os.getcwd()
    path_csv = os.path.join(project_path, "CSV", "adam_features.csv") 
    df_features.to_csv(path_csv)
    stop = 0


    # CLASSIFICATION

    X, y = dc.getXyfromCSV(path_csv)



if __name__ == "__main__":
    main()
