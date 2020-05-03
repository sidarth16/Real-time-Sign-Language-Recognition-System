#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import math
import sys
import os


#  Here initially the required libraries are initially imported to program like **OpenCV (cv2), NumPy , Math , Sys , os**
#  

# In[1]:



def generate_name(length = 3):
    char_list = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
    name = []
    for i in range(length):
        name.append(char_list[np.random.randint(len(char_list))])
    return "".join(name)

def getNewLabel(name,value = 'Dataset_1'):
    current_directory = os.getcwd()
    dataset_directory = os.path.join(current_directory,value)
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    final_directory = os.path.join(dataset_directory, name)
    subdirectory_1 = os.path.join(final_directory, 'Original')
    subdirectory_2 = os.path.join(final_directory, 'skin')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        os.makedirs(subdirectory_1)
        os.makedirs(subdirectory_2)
    return [subdirectory_1 , subdirectory_2 ]


# 
#  Here , the function methods **generate_name()** and **getNewLabel()** are defined where , *generate_name()* randomly generates name of the image file to be stored and *getNewLabel()* returs the directory path of the image to be stored . Here the library *os* helps in getting the cuurent directory path using *os.getcwd()* through which we traverse and create new directory through *os.path.join()* , Thus returning the new directory path
# 
# 

# In[2]:


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    start , end = 200 , 450 

    cv2.rectangle(img, (start+500,start-150), (end+500,end-150), (102,185,255),5)
    crop_img = img[start-150:end-150, start+500:end+500]
    image = crop_img
    
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
    skin_img=skinYCrCb

    all_img = np.hstack((crop_img,skin_img))
    cv2.imshow('WebCam', img)
    cv2.imshow('All', all_img)

    k = cv2.waitKey(10)

    if k == 27:   #esc key
        break
        
    elif (((k>=65 and k<=90)) or ((k>=97) and (k<=122))):

        p=chr(k) #Ascii to char   
        newlabel = p
        original_dir = getNewLabel(newlabel)
        
        number = len(os.listdir(original_dir[0])) + 1
        print('label :'+p+' captured! ,  count = '+str(number))

        original_img_path = os.path.join(original_dir[0] ,generate_name()+'.jpg')
        skin_img_path = os.path.join(original_dir[1], generate_name()+'.jpg')
    
        cv2.imwrite(original_img_path,crop_img)
        cv2.imwrite(skin_img_path,skin_img)
    
cap.release()
cv2.destroyAllWindows()


# 
# Here we initally set the *frame width and height dimensions* of the input Webcam Feed frame. the Bounding Box is then marked down by drawing a rectangle in the frame using *cv2.rectangle()* . 
# 
# the image only inside the bounding box is then selected and stored as **crop_img**.
# The image is then passed through filter which converts the image to YCrCb format and selects only those pixels within the specified skin color range , thus removing other noise objects from the frame. This image is saved as **skin_img**
# 
# The cropped image then is displayed to the user using *cv2.imshow()* , on pressing any key in the keyboard , the respective image is then saved under the folder of the key pressed .
# 
# *Example :*  if the Key A is pressed then a folder named "A" is created and the cropped image without skin filter is saved under the "Original" directory and skin filtered cropped image is saved under directory named "skin".
# If the directory already exists , then the folder is updated and appended with the new data of images
# 

# In[ ]:




