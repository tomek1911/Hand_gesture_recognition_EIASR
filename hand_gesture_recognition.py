import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import string
import csv

#project directory, directory of dataset

dir = os.getcwd()
dataset_dir = os.path.join(dir, "Data")
plot_dir = os.path.join(dir, "Plots and graphs")


img_list = os.listdir(dataset_dir) 
print ("Dataset contains: ",len(img_list), "images.")

#analyse dataset
#create csv file with paths and class labels

gestures_dict = dict.fromkeys(string.ascii_uppercase, 0)
dataset = []

for elem in img_list:
    num = gestures_dict[elem[0]]
    gestures_dict[elem[0]]= num + 1
    dataset.append([elem, elem[0]])

cw = csv.writer(open("gestures_dataset.csv",'w'), delimiter =',')

for elem in dataset:
    cw.writerow(list(elem))

#print number of examples for each sign 

print("Dataset stats for gestures:")
for key, value in gestures_dict.items():
    if value >0:
       print(key,":",value)

gestDict_classes = gestures_dict.keys()
gestDict_counts = gestures_dict.values()

y_classesCount = np.arange(len(gestDict_classes))

# create bar chart - count samples for classes

plt.bar(y_classesCount, gestDict_counts, align='center', alpha=0.5)
plt.xticks(y_classesCount, gestDict_classes)
plt.ylabel('Count')
plt.xlabel('Sign')
plt.title('Number of samples for each class')
plt.savefig(plot_dir + "//barPlotSignCount.png")

#create summary: image samples with labels 
#one image for each class - images are downsized 

aslAphabet = list(string.ascii_uppercase)
aslAphabet.remove('J')
aslAphabet.remove('Z')
sampleImages = []

for sign in aslAphabet:
   first_elem = next((x for x in dataset if x[1] == sign),None)
   img = cv2.imread(dataset_dir+"//"+first_elem[0])
   img_resized = cv2.resize(img,(60,80),interpolation=cv2.INTER_AREA)
   img_resizedRGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
   sampleImages.append(img_resizedRGB)    

fig=plt.figure(figsize=(8, 8))
plt.suptitle("Examples from our dataset for each of signs")
columns = 4
rows = 6
plt.subplots_adjust(wspace=0.0, hspace = 0.4)
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i, xticks=[], yticks=[], title=aslAphabet[i-1])
    plt.imshow(sampleImages[i-1])
plt.savefig(plot_dir + "//datasetSample.png")


















