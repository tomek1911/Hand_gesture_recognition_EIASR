import os
import numpy as np
import cv2
import csv
import pandas as pd

from enum import Enum

from data_preprocessing import DataPreprocessing as dp
from feature_extraction import FeatureExtraction as fe
from data_classification import DataClassification as dc
import camera

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
      
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
        # else:
            # print("List of images to load is empty or images are already loaded")

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
    
    @staticmethod
    def csvToListOfLists(filename, folder):

        myfile = filename + ".csv"
        path = os.path.join(os.getcwd(), folder, myfile)          
         
        try:
            with open(path, 'r') as read_obj:    
                csv_reader = csv.reader(read_obj)
                return list(csv_reader)
        except IOError:
            print("I/O error")    

    @staticmethod
    def manualTresholdTester(image, colorspace = cv2.COLOR_BGR2YCrCb, slider_maxCh1 = 255, slider_maxCh2 = 255, slider_maxCh3 = 255):
    
        title_window = "Universal Thresholding Manual Sandbox"        
       
        image = cv2.cvtColor(image, colorspace)

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

        cv2.createTrackbar(tNameCh1_low, title_window , 0, slider_maxCh1, on_trackbar)
        cv2.createTrackbar(tNameCh1_high, title_window , 0, slider_maxCh1, on_trackbar)
        cv2.createTrackbar(tNameCh2_low, title_window , 0, slider_maxCh2, on_trackbar)
        cv2.createTrackbar(tNameCh2_high, title_window , 0, slider_maxCh2, on_trackbar)
        cv2.createTrackbar(tNameCh3_low, title_window , 0, slider_maxCh3, on_trackbar)
        cv2.createTrackbar(tNameCh3_high, title_window , 0, slider_maxCh3, on_trackbar)

        #Inital tresholds 
        cv2.setTrackbarPos(tNameCh1_low, title_window,50)
        cv2.setTrackbarPos(tNameCh1_high, title_window,255)
        cv2.setTrackbarPos(tNameCh2_low, title_window,140)
        cv2.setTrackbarPos(tNameCh2_high, title_window,180)
        cv2.setTrackbarPos(tNameCh3_low, title_window,60)
        cv2.setTrackbarPos(tNameCh3_high, title_window,130)

        cv2.waitKey(0)
        cv2.destroyAllWindows()     

    def __init__(self, datasetFolder):

        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, datasetFolder) 
        self.getImagesToLoad()
        self.imagesList_cv = []
        self.dataset_array = []


def handDetRec(frame, key):

    if key == ord('r'):
        #TODO normalize image size to square - center shape in ROI
        # img_rgb = dp.resizeImage(frame, 120)
        img = dp.tresholdImageYCBCR(frame)
        img = dp.morphologicFiltering(img, (5, 5))

        return img
    else:
        return None

        contour = dp.filterContours(img)
        img_hand_binary = np.zeros((160, 120), np.uint8)
        if len(contour) == 0:
            return None
        else:
            cv2.fillPoly(img_hand_binary, pts=[contour], color=(255))
            img_hand = cv2.bitwise_and(img_rgb, img_rgb, mask=img_hand_binary)
            # dict_adam = fe.getAdamFeatures(contour,img_hand_binary)
            hog_vec, hog_img = fe.getHog(img_hand, _multichannel=True)
            return hog_img
    # cv2.imwrite("/home/tomek/Projects/Hand_gesture_recognition_EIASR/Processed/LoopTest/img.png", hog_img)

 


def main():

    # cam = camera.Camera(method=handDetRec, args=[None, None])
    # cam.initCameraLoop()

    # return
    
    print("ASL Hand Gestures Recgonition - Initialize")
    #Initialisation
    dLoader_obj = DataLoader("Data") # load dataset images directories
    dLoader_obj.describeLoadedData() # read labels and images id 
    print(*dLoader_obj.dataset_array[0:5], sep="\n")  # print labels and images id - first 5   

    #Test tresholds for data_processing
    dLoader_obj.loadImagesCv(1) # load one of images 
    image = dLoader_obj.imagesList_cv[0]
    #DataLoader.manualTresholdTester(image) - uncomment

    # assign output folder path for processed images
    outPut_dir = os.path.join(dLoader_obj.project_dir,"Processed") 

    ##########################################################################################################################
    #
    # DEVELOMPENT STAGES - workflow - read images from file, process, save to file 
    #
    ##########################################################################################################################
    #RESIZE
    # print("Resize original datset images:")
    # printProgressBar(0, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    # for i in range(0,len(dLoader_obj.imagesList_dir)):
    #     img = dLoader_obj.loadImageCv(i)
    #     resized_img = dp.resizeImage(img,120)
    #     dp.save_image(resized_img,dLoader_obj.dataset_array[i],"ResizedImages",outPut_dir,95,"png")
    #     printProgressBar(i + 1, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    ##########################################################################################################################
    #THRESHOLD                
    
    print("Thresholding - YCbCr colorspace based:")
    dLoader_obj_resized = DataLoader("Processed/ResizedImages")   
    dLoader_obj_resized.describeLoadedDataPNG()
    dLoader_obj_resized.loadImagesCv()

    printProgressBar(0, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)
    # tresholdedImagesList = []

    for i in range(0,len(dLoader_obj_resized.imagesList_dir)):
        img = dLoader_obj_resized.loadImageCv(i)
        tresholdedImage = dp.tresholdImageYCBCR(img)
        dp.save_image3(tresholdedImage,dLoader_obj_resized.dataset_array[i],"TresholdedImages",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    #########################################################################################################################
    #RESULTS PREVIEW - create figures of concatenated images - 2520x1440px - 189 images
    
    dLoader_obj_tresh = DataLoader("Processed/TresholdedImages")   
    dLoader_obj_tresh.describeLoadedDataPNG()
    dLoader_obj_tresh.loadImagesCv()
    
    # tresholdedImagesList = dLoader_obj_tresh.imagesList_cv         
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
 
    print("Morphologic filtering - opend and close:")
    printProgressBar(0, len(dLoader_obj.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_tresh.imagesList_dir)):
        img = dLoader_obj_tresh.loadImageCv(i)
        morph_img = dp.morphologicFiltering(img, (5,5))
        dp.save_image3(morph_img,dLoader_obj_tresh.dataset_array[i],"MorphFilter",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_tresh.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    #########################################################################################################################
    #CONTOURS

    print("Filtering contours - chose hand contour:")
    dLoader_obj_binary = DataLoader("Processed/MorphFilter")   
    dLoader_obj_binary.describeLoadedDataPNG()
    dLoader_obj_binary.loadImagesCv()
 
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
    #Center and normalize 
    # segmented hand - mask

    print("Center and normalize image (square shape):")
    dLoader_obj_segmented = DataLoader("Processed/Contours")   
    dLoader_obj_segmented.describeLoadedDataPNG()
    dLoader_obj_binary.loadImagesCv()    

    printProgressBar(0, len(dLoader_obj_segmented.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_segmented.imagesList_dir)):
        img = dLoader_obj_segmented.loadImageCvGray(i)

        img_out = dp.centerToSquare(img, contoursList[i], margin=16)

        dp.save_image3(img_out,dLoader_obj_segmented.dataset_array[i],"Squares",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_segmented.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)  
     
    #########################################################################################################################
    #Center and normalize 
    # rgb_images

    print("Center and normalize image - rgb images:")
    dLoader_obj_resized = DataLoader("Processed/ResizedImages")   
    dLoader_obj_resized.describeLoadedDataPNG()
    dLoader_obj_resized.loadImagesCv() 

    printProgressBar(0, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_resized.imagesList_dir)):
        img = dLoader_obj_resized.loadImageCv(i)

        img_out = dp.centerToSquare(img, contoursList[i], margin=16)

        dp.save_image3(img_out,dLoader_obj_resized.dataset_array[i],"SquaresRGB",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_resized.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)  


    #########################################################################################################################
    #RGB coutout with mask
    
    print("Cuting out rgb images with segmented hand mask:")
    dLoader_obj_segmented = DataLoader("Processed/Contours")   
    dLoader_obj_segmented.describeLoadedDataPNG()
    dLoader_obj_binary.loadImagesCv()

    printProgressBar(0, len(dLoader_obj_segmented.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0,len(dLoader_obj_segmented.imagesList_dir)):
        img = dLoader_obj_resized.loadImageCv(i)
        mask_img = dLoader_obj_segmented.loadImageCvGray(i) # mask       
        res = cv2.bitwise_and(img,img,mask = mask_img)
        dp.save_image3(res,dLoader_obj_segmented.dataset_array[i],"CutoutRGB",outPut_dir,95,"png")
        printProgressBar(i + 1, len(dLoader_obj_segmented.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)   
      

    #########################################################################################################################
    #FEATURES I - handmade

    print("Features I - handmade:")
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

    #add features I to dataframe, save them to csv file
    df_features = pd.DataFrame(featuresList)
    project_path = os.getcwd()
    path_csv = os.path.join(project_path, "CSV", "adam_features.csv") 
    df_features.to_csv(path_csv)

    #########################################################################################################################
    #FEATURES IIa - hog (!long calc)
    
    # print(" Features II - hog (grayscale images):")
    # hogFeaturesList = []    
    # hogImgLabels = []
    # printProgressBar(0, len(dLoader_obj_cont.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    # for i in range(0,len(dLoader_obj_cont.imagesList_dir)):
    #     img = dLoader_obj_cont.loadImageCvGray(i)
    #     hog_vec, hog_img = fe.getHog(img,_multichannel=False)
    #     hogFeaturesList.append(hog_vec)
    #     hogImgLabels.append(dLoader_obj_cont.dataset_array[i][0])
    #     dp.save_image3(hog_img,dLoader_obj_cont.dataset_array[i],"Hog",outPut_dir,95,"png")
    #     printProgressBar(i + 1, len(dLoader_obj_cont.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)   

    # #add features II to dataframe, save them to csv file
    # df_features2 = pd.DataFrame(hogFeaturesList)
    # project_path = os.getcwd()
    # path_csv_hog = os.path.join(project_path, "CSV", "hog_features.csv") 
    # df_features2.to_csv(path_csv_hog)


    #########################################################################################################################
    #FEATURES IIb - hog RGB (!long calc)

    # dLoader_obj_rgbmasked = DataLoader("Processed/CutoutRGB")   
    # dLoader_obj_rgbmasked.describeLoadedDataPNG()
    # dLoader_obj_rgbmasked.loadImagesCv() #wczytaj wszystkie zdjecia z folderu 

    # print("Features II - hog (rgb images):")
    # hogRGBFeaturesList = []    
    # hogImgLabels = []
    # printProgressBar(0, len(dLoader_obj_rgbmasked.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)

    # for i in range(0,len(dLoader_obj_rgbmasked.imagesList_dir)):
    #     img = dLoader_obj_rgbmasked.loadImageCv(i)
    #     hog_vec, hog_img = fe.getHog(img,_multichannel=True)
    #     hogRGBFeaturesList.append(hog_vec)
    #     hogImgLabels.append(dLoader_obj_rgbmasked.dataset_array[i][0])
    #     dp.save_image3(hog_img,dLoader_obj_rgbmasked.dataset_array[i],"HogRGB",outPut_dir,95,"png")
    #     printProgressBar(i + 1, len(dLoader_obj_rgbmasked.imagesList_dir), prefix = 'Progress:', suffix = 'Complete', length = 50)   

    # #add features IIb to dataframe, save them to csv file
    # df_features2b = pd.DataFrame(hogRGBFeaturesList)
    # project_path = os.getcwd()
    # path_csv_hog_rgb = os.path.join(project_path, "CSV", "hog_rgb_features.csv") 
    # df_features2b.to_csv(path_csv_hog_rgb)        

    #hog calculation takes log for dataset so we read features from file
    hogImgLabels = []    
    for i in range(0,len(dLoader_obj_cont.imagesList_dir)):
         hogImgLabels.append(dLoader_obj_cont.dataset_array[i][0])

    useRGB_hog_features = True
    featuresFile = ''

    if useRGB_hog_features:
        featuresFile = 'hog_rgb_features'
        print("Using RGB HOG features")
    else:
        featuresFile = 'hog_features'
        print("Using grayscale HOG features")

    hogFeaturesList = DataLoader.csvToListOfLists(featuresFile, "CSV")
    
    # remove list with feature ids, and column with vector id
    del hogFeaturesList[0]
    for vector in hogFeaturesList:
        del vector[0]

    #########################################################################################################################
    # CLASSIFICATION
    # k-NN classifier

    print("k-NN classifier, features I:")
    X, y = dc.getXyfromCSV(path_csv)
    dc.fitKnn(X,y,print_res=True)
    
    # SVM classifier (<1min)
    print("SVM classifier, features II:")
    dc.fitSVM(hogFeaturesList, hogImgLabels, print_res=True, confusionMatrix=True) 


if __name__ == "__main__":
    main()
