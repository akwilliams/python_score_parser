import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

#Initialize gloabal arguments
sources = []
types = ["c_clef","f_clef","g_clef","flat","natural","sharp"]

#load individual inverted images for comparison 
def load_img(folder,file):
    path = folder+file
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img_blr=cv2.GaussianBlur(img,(9,9),2)
    th,img_th=cv2.threshold(img_blr,150,255,cv2.THRESH_BINARY_INV)
    return img_th
#load all comparison sources
def load_srcs():
    for i in range(len(types)):
        sources.append([])
        path = "source/"+types[i]+"/"
        for j in range(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))])):
            file = 'img_'+str(j)+'.png'
            if file in os.listdir(path):
                sources[i].append(load_img(path,file))
            
 




load_srcs()
temp = cv2.imread('source/scores/img_0.png',cv2.IMREAD_GRAYSCALE)
current = cv2.GaussianBlur(temp,(37,37),75)
th,current=cv2.threshold(current,125,255,cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(current, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    current=cv2.drawContours(current,[box],0,(150,150,150),3)

#cnt=contours[1]
#cv2.drawContours(current,contours,0,(0,255,0),3)

cv2.imshow('img',current)
cv2.waitKey(0)
cv2.destroyAllWindows()