import cv2
import matplotlib
import numpy as np
import os
import string
import csv


dir = os.getcwd()
dataset_dir = os.path.join(dir, "Data")

img_list = os.listdir(dataset_dir) 
print ("Dataset contains: ",len(img_list), "images.")


gestures_dict = dict.fromkeys(string.ascii_uppercase, 0)

dataset = []

for elem in img_list:
    num = gestures_dict[elem[0]]
    gestures_dict[elem[0]]= num + 1
    dataset.append([elem, elem[0]])

cw = csv.writer(open("gestures_dataset.csv",'w'), delimiter =',')

for elem in dataset:
    cw.writerow(list(elem))

print("Dataset stats for gestures:")
for key, value in gestures_dict.items():
    if value >0:
       print(key,":",value)










