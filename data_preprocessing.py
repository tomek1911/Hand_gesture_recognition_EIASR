import cv2
import numpy as np
import os
from enum import Enum
from dataclasses import dataclass
    

class SkinSegmentation:   
# HSV: 0.23 < S < 0.68, 0 < H < 50
# YCBCR: 135< Cr < 180, 85 < Cb < 135, Y > 80
# YUV: 65 < Y < 170, 85 < U < 140, 85 < V < 160 

        @dataclass
        # HSV[0-180,0-255,0-255]       
        class HSV_Threshold:
            S_min: int = 59
            S_max: int = 173
            H_min: int = 0
            H_max: int = 50
            V_min: int = 0
            V_max: int = 255

        @dataclass
        class YCBCR_Threshold:
            Y_min: int = 50 #Blue channel [0]
            Y_max: int = 255
            Cr_min: int = 140 #Green channel [1]
            Cr_max: int = 180
            Cb_min: int = 95 # Red channel [2]
            Cb_max: int = 125


        @dataclass
        class YUV_Threshold:
            Y_min: int = 65
            Y_max: int = 170
            U_min: int = 85
            U_max: int = 140
            V_min: int = 85
            V_max: int = 160

class DataPreprocessing:    
    """Class provides tools to preprocess images."""
    pass   

    imagesList_toProcess = []
    imagesDetails = []
    resized_images = []
    output_dir = ""

    COLORSPACE = Enum('Colorspace', 'HSV YUV YCBCR')   

    #resize images - based on length of the shorter edge
    def resizeImages(self,shortEdgeLength): 

        resized_images = [] 

        if self.imagesList_toProcess:
            rows,cols,_ = self.imagesList_toProcess[0].shape
            shortEd = min(rows,cols)
            longEd = max(rows, cols) 
            # aspect = longEd / shortEd
            resizeRatio = shortEd / shortEdgeLength

            newLongEdgeLegth = round (longEd / resizeRatio)            

            for img in self.imagesList_toProcess:
                img_resized = cv2.resize(img,(shortEdgeLength,newLongEdgeLegth),interpolation=cv2.INTER_AREA)        
                resized_images.append(img_resized)
        
        return resized_images

   
        
    def save_processed_images(self, processedImages, folderName, quality):
        for img, imgData in zip(processedImages, self.imagesDetails):
            filename = imgData[1]+"_"+imgData[2]+"_"+imgData[3][0]+".jpg"
            imwrite_dir = os.path.join(self.output_dir,folderName)
            imwrite_path = os.path.join(imwrite_dir, filename)
            
            if os.path.isdir(imwrite_dir):  
                cv2.imwrite(imwrite_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            else:
                os.makedirs(imwrite_dir)
                cv2.imwrite(imwrite_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    @staticmethod
    def save_image(image, imgData, folderName, output_dir, quality=80):
        sufix = ''
        if imgData[3] == "Oleksandr_2":
            sufix = '2'
        filename = imgData[1]+"_"+imgData[2]+"_"+imgData[3][0]+sufix+".jpg"

        imwrite_dir = os.path.join(output_dir, folderName)
        imwrite_path = os.path.join(imwrite_dir, filename)
        if not os.path.isdir(imwrite_dir):
            os.makedirs(imwrite_dir)     
        cv2.imwrite(imwrite_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    @staticmethod
    def save_image2(image, imgName, folderName, output_dir, quality=80):

        filename = imgName+".jpg"
        imwrite_dir = os.path.join(output_dir, folderName)
        imwrite_path = os.path.join(imwrite_dir, filename)
        if not os.path.isdir(imwrite_dir):
            os.makedirs(imwrite_dir)     
        cv2.imwrite(imwrite_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
    @staticmethod 
    def resizeImage(image, shortEdgeLength):
        rows,cols,_ = image.shape

        shortEdge = min(rows,cols)
        longEdge = max(rows, cols) 
        resizeRatio = shortEdge / shortEdgeLength
        newLongEdgeLegth = round (longEdge / resizeRatio) 
        if newLongEdgeLegth > longEdge:
            img_resized = cv2.resize(image,(shortEdgeLength,newLongEdgeLegth),interpolation=cv2.INTER_CUBIC) #enlarge 
        else:
            img_resized = cv2.resize(image,(shortEdgeLength,newLongEdgeLegth),interpolation=cv2.INTER_AREA) #shrink     
        return img_resized 


    def skinDetection(self, colorSpace, image):

        skinSeg_obj = SkinSegmentation()

        if colorSpace.value == self.COLORSPACE.HSV.value:
            
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
            thr = skinSeg_obj.HSV_Threshold() # hsvTresholdValues
            lowerBound = (thr.H_min, thr.S_min, thr.V_min)
            upperBound = (thr.H_max, thr.S_max, thr.V_max)
            img_hsv_tresholded = cv2.inRange(img_hsv, lowerBound,  upperBound)
            return img_hsv, img_hsv_tresholded

            # cv2.imshow("HSV_img", img_hsv) 
            # cv2.imshow("HSV_tresh", img_hsv_tresholded) 
            # cv2.waitKey(0)  
        
        elif colorSpace.value == self.COLORSPACE.YUV.value:
            img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            thr = skinSeg_obj.YUV_Threshold() # yuvTresholdValues
            lowerBound = (thr.Y_min, thr.U_min, thr.V_min)
            upperBound = (thr.Y_max, thr.U_max, thr.V_max)
            img_cvt_tresholded = cv2.inRange(img_cvt, lowerBound,  upperBound)
            return img_cvt, img_cvt_tresholded

           
        elif colorSpace.value == self.COLORSPACE.YCBCR.value:
            img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            thr = skinSeg_obj.YCBCR_Threshold() # yuvTresholdValues
            lowerBound = (thr.Y_min, thr.Cr_min, thr.Cb_min)
            upperBound = (thr.Y_max, thr.Cr_max, thr.Cb_max)
            img_cvt_tresholded = cv2.inRange(img_cvt, lowerBound,  upperBound)
            return img_cvt, img_cvt_tresholded

    @staticmethod
    def tresholdImageYCBCR(imageBGR):

        Y_min = 30 #Blue channel [0]
        Y_max = 255
        Cr_min = 139 #Green channel [1]
        Cr_max = 180
        Cb_min = 93 # Red channel [2]
        Cb_max = 127

        img_cvt = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YCrCb)

        lowerBound = (Y_min, Cr_min, Cb_min)
        upperBound = (Y_max, Cr_max, Cb_max)
        
        img_cvt_tresholded = cv2.inRange(img_cvt, lowerBound, upperBound)

        return img_cvt_tresholded       
    
    @staticmethod
    def morphologicFiltering(img,size):
        kernel = np.ones(size, np.uint8)
        img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img_open_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
        return img_open_close
   
      


    def __init__(self, imagesList_cv=None, imagesDetails=None, output_dir=None):
        if imagesList_cv is None:
            imagesList_cv = []
        else:
            self.imagesList_toProcess = imagesList_cv

        if imagesDetails is None:
            imagesDetails = []
        else:
            self.imagesDetails = imagesDetails

        if output_dir is None:
            output_dir = []
        else:
            self.output_dir = output_dir

  


        
            


