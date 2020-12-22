import cv2
import numpy as np
import math

class FeatureExtraction:
    """Class provides tools to extract features from images."""
    pass

    @staticmethod
    def getAdamFeatures(binary_image):
        features_dict = {}

        contours, _ = cv2.findContours(binary_image,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        # Unpack an array of contours
        cnt = contours[0] #TODO - dodac filtrowanie konturow - trzeba wybrac najwiekszy - pewnie nalezy do reki - dodatkowy warunek np minimum 25% powierzchni zdjecia
        # Moments
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
    def featureDict2x(feat_dict):
        x = []
        for feat in list(feat_dict.values()):
            x.append(feat)
        return x

    def __init__(self):
        pass