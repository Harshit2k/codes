#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import sys


# In[2]:


img_path = r'C:\Users\harsh\Documents\Python_Data_Sets\natural-images\natural_images\person\person_0185.jpg'
cascade_path = r'C:\Users\harsh\Documents\Python_Data_Sets\haarcascades(face_detection)\haarcascade_frontalface_default.xml'


# In[3]:


facecascade=cv2.CascadeClassifier(cascade_path)


# In[4]:


img=cv2.imread(img_path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[5]:


face=facecascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
print('Total no of faces found:',len(face))


# In[6]:


for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) 
    plt.imshow(img)  
  

