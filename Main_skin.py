#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import pandas as pd
import math
import sys
import os
import tensorflow as tf 
from keras.models import load_model
from datetime import datetime

import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
winname="Brightness & Contrast"
bright_val = 0
contrast_val = 0

try:
    def on_trackbar(val):
        pass
 
    cv2.namedWindow(winname , cv2.WINDOW_GUI_EXPANDED )        # Create a named window
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("brightness", winname ,0, 100, on_trackbar )
    cv2.createTrackbar("Contrast", winname , 0, 100, on_trackbar )

    def on_trackbar(val):
        pass
 
    while(True):  
        value1 = cv2.getTrackbarPos("brightness", winname)
        value2 = cv2.getTrackbarPos("Contrast", winname)
        
        ret, img = cap.read()
        img = cv2.flip(img,1)
        bright_val = value1+0.5
        contrast_val = value2*0.25 + 1
        img1 = cv2.add(img,np.array([bright_val]))
        img1 = cv2.multiply(img1,np.array([contrast_val]))

        cv2.imshow(winname,img1)
        k = cv2.waitKey(10)
        if k == 27:
            break
finally :
    cap.release()
cv2.destroyAllWindows()

sign_chart = cv2.imread("guide_final1.png")
cap = cv2.VideoCapture(0)

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume'  ,1)
voice_zira = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
voice_david = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"

engine.setProperty('voice', voice_zira)
def speak(string):
    engine.say(string)
    engine.runAndWait()

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

color1 = (59, 185, 246)
color2 = (95, 105, 246)
color3 = (200, 40, 96)

alpha = 0.4

img_height = 480
img_width  =  640
  
main_c = (510 , 130 , 110)

num= 0 

def write_to_file(msg) :
    t = str(datetime.now())
    name_of_file = t[0:len(t)-7].replace(" ","_").replace(":","_").replace("-","_")
    current_directory = os.getcwd()
    save_directory = os.path.join(current_directory,'saved_files')
    completeName = os.path.join(save_directory, name_of_file+".txt")
    file = open(completeName, "w")
    file.write(msg)
    file.close()
    print("file:"+name_of_file+".txt saved successfully")
    
def image_resize_for_predict(image, height = 45, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (height,height), interpolation = inter) #
    return resized


def image_resize(image, height = 230, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (190 , 190), interpolation = inter) #(height,height)
    return resized

def rect_points_from_cirle(x,y,r) : 
    x1,y1 = (x - r) , (y - r )
    x2,y2 = (x + r) , (y + r )
    return (x1 , y1 , x2 , y2 )

def str_partition(string , limit ):
    s=[]
    t=""
    for i in string:
        t=t+i
        if(len(t)==limit):
            s.append(t)   
            t=""
    if(len(t)>0):
        s.append(t)
    if(len(s)<5):
        t=5-len(s)
        for i in range(t):
            s.append(" ")
    return(s)

model = load_model('CNN_model.h5')
encoding_chart = pd.read_csv('label_encoded.csv')
encoding_values = encoding_chart['Encoded'].values
encoding_labels = encoding_chart['Label'].values
int_to_label = dict(zip(encoding_values,encoding_labels))

font = cv2.FONT_HERSHEY_DUPLEX

history = list()
counts = dict()
history_length = 15
threshold = 0.9

sentence_raw = list()


m_rect   = rect_points_from_cirle( main_c[0] , main_c[1] , main_c[2] )    #Main

voice_flag=1
pause = False

try :
    while(True):
        
        ret, img = cap.read()
        img = cv2.flip(img,1)
        img = cv2.add(img,np.array([bright_val]))
        img = cv2.multiply(img,np.array([contrast_val]))

        alpha_layer = img.copy()
        source = img.copy()

        crop_img = source[ m_rect[1]:m_rect[3] , m_rect[0]:m_rect[2]] 
        image = crop_img
        imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
        skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
        skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
        crop_img = skinYCrCb
        
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        _, thresh1 = cv2.threshold(blurred, 135, 255 , cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernal = np.ones((10,10),np.uint8)
        erosion = cv2.erode(thresh1,kernal,iterations=1)
        dilation = cv2.dilate(erosion,kernal,iterations=1)
        
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

        hull = cv2.convexHull(cnt, returnPoints=False)

        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)
   
        label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'None', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'Space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    
        
        resized_img = image_resize(crop_img)       # if predict using original img
        resized_thresh = image_resize(dilation_rgb)   # if predict using threshold img
        resized_draw = image_resize(drawing)         # if predict using drawing
    
        resized_for_predict = image_resize_for_predict(resized_img)
                
        predicted = model.predict(np.array([resized_for_predict]))
        predicted_char = int_to_label[np.argmax(predicted)]
        
        acc = np.max(predicted) #output[maxpos] #np.max(predicted)   #accuracy
        acc = str(acc*100)[:7]
       
   
        kt_height , kt_width = 47 , 1350
        kernal_top = np.ones((kt_height , kt_width,3),np.uint8)
    
        kl_height , kl_width = 480,310
        kernal_left = np.ones((kl_height , kl_width,3),np.uint8)
    
        kr_height , kr_width = 480,400
        kernal_right = np.ones((kr_height , kr_width,3),np.uint8)
    
        kb_height , kb_width = 180,1350
        kernal_bottom = np.ones((kb_height , kb_width,3),np.uint8)
    
    
        #Top kernal
        cv2.putText(kernal_top, "Real-Time Sign Language Translator" , (330,35),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(150,255,255), thickness=2)
    
        #Bottom kernal
        kernal_bottom[25:170 , 100:1281] = sign_chart #[ 25 : 145+25=170 , 100 : 100+1181=1281 ]
    
        #Middle kernal
        main_rect = cv2.rectangle(alpha_layer ,  (m_rect[0],m_rect[1]), (m_rect[2],m_rect[3])  , color1, -1)
        cv2.addWeighted(alpha_layer, alpha, img, 1 - alpha,0, img)
    
        #Left kernal
        kernal_left[10:200  , 100:290] = resized_img#thresh
        kernal_left[280:470 , 100:290] = resized_draw
    
        #Right kernal
        thresh_rect = cv2.rectangle(kernal_left ,  (100,10), (290,200)  , (120,180,200), 4)
        draw_rect = cv2.rectangle(kernal_left ,  (100,280), (290,470)  , (120,180,200), 4)
   
        text_rect = cv2.rectangle(kernal_right ,  (10,10), (385 , 200)  , (120,180,200), 4)
        msg_rect  = cv2.rectangle(kernal_right ,  (10,225), (385 , 400)  , (120,180,200), 4)

        cv2.putText(kernal_right, ' A : Audio Playback   D : Delete', (12,435),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.65, color=(255,255,255), thickness=1)
        cv2.putText(kernal_right, ' X : Delete All   S : Save to file ', (12,470),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.65, color=(255,255,255), thickness=1)
    
        cv2.putText(kernal_right, 'Predicted Alphabet : ', (25,40),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.9, color=(255,255,255), thickness=2)
    
    
        if(len(history)>=history_length):
            # print("history full : ")# print(history , counts)
            keys = list(counts.keys())     #count = {A:7 , B:1 , C:5 , D:2} total 15 values as per history_length
            values = list(counts.values())
            arg = np.argmax(values)   
            if(values[arg]>threshold*history_length):
                speak(keys[arg])
                if(keys[arg] == 'Space'):
                    sentence_raw.append(" ")
                    speak(sentence)
                else:
                    sentence_raw.append(keys[arg])
            counts.clear()
            history.clear()
          #  print("history , count cleared")#  print(history , counts)
        if(predicted_char != 'None' ):
         #   print("predicted char : "+predicted_char)#   print(history , counts)
            if(len(sentence_raw) > 0):
                    if(predicted_char == sentence_raw[-1]):
                        history_length = 25
                    else :
                        history_length = 15
            history.append(predicted_char)
            if(predicted_char in counts):
                counts[predicted_char]+=1
            else:
                counts[predicted_char]=1
            textsize = cv2.getTextSize(predicted_char, font, 6,7)[0]
            
            if(predicted_char == "Space"):
                cv2.putText(kernal_right, predicted_char, (75,135),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 3, color=(0,255,0), thickness=3)
            else:
                cv2.putText(kernal_right, predicted_char, (160,135),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 3, color=(0,255,0), thickness=3)
            cv2.putText(kernal_right, "Accuracy :"+acc , (25,180),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(150,255,0), thickness=1)
    
    
    
        else:
            cv2.putText(kernal_right, predicted_char, (75,135),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 3, color=(0,255,0), thickness=3)
            cv2.putText(kernal_right, "Accuracy :"+acc , (25,180),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(150,255,0), thickness=1)

        scribble = "".join(sentence_raw)
        sentence = scribble 
        sentencesize = cv2.getTextSize(sentence, font, 1,2)[0]
 
        s = str_partition(scribble , 18 )
        s0_size = cv2.getTextSize(s[0], font, 1,2)[0]
        s1_size = cv2.getTextSize(s[1], font, 1,2)[0]
        s2_size = cv2.getTextSize(s[2], font, 1,2)[0]
        s3_size = cv2.getTextSize(s[3], font, 1,2)[0]
        s4_size = cv2.getTextSize(s[4], font, 1,2)[0]
    
        cv2.putText(kernal_right, s[0] , (int((kr_width - s0_size[0])/2),255),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(255,255,255), thickness=2)
        cv2.putText(kernal_right, s[1] , (int((kr_width - s1_size[0])/2),285),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(255,255,255), thickness=2)
        cv2.putText(kernal_right, s[2] , (int((kr_width - s2_size[0])/2),315),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(255,255,255), thickness=2)
        cv2.putText(kernal_right, s[3] , (int((kr_width - s3_size[0])/2),345),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(255,255,255), thickness=2)
        cv2.putText(kernal_right, s[4] , (int((kr_width - s4_size[0])/2),375),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color=(255,255,255), thickness=2)
    
    
       
    
        kernel_centre = np.hstack((kernal_left ,img , kernal_right))
        all_kernel    = np.vstack((kernal_top ,kernel_centre , kernal_bottom))
    

        if voice_flag==1:
            speak('Hello Iam Hazel , Welcome to Sign Language converter')
            voice_flag=0
    
        window = "ASL Translator"
        cv2.namedWindow(window , cv2.WINDOW_GUI_EXPANDED )        # Create a named window
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
   
        cv2.imshow(window, all_kernel)
    
        if(pause):
            history.clear() 
            counts.clear()

        k = cv2.waitKey(10)
        if k == 27:
            break
        elif k == ord('x') or k == ord('X'):
            sentence_raw.clear()
            counts.clear()
            history.clear()
    
        elif k==ord('p') or k==ord('P'):
            pause = True
            
        
        elif k==ord('s') or k==ord('S'):
            write_to_file(sentence)
        
        elif k == ord('d') or k == ord('D'):
            if(len(sentence_raw)>0):
                sentence_raw.pop()
                counts.clear()
                history.clear()
        
        elif k==ord('n') or k==ord('N'):
            num=1
            
        elif k==ord('a') or k==ord('A') :
            speak(sentence)

    
finally :
    cap.release()

cv2.destroyAllWindows()

