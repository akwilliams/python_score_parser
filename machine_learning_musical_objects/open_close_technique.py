import cv2
import os
import random
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt


img=cv2.imread('source/scores/img_60.png',cv2.IMREAD_GRAYSCALE)
height,width=img.shape[:2]
filter_arg=int(width*0.5)
filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
img_filtered=cv2.GaussianBlur(img,(filter_arg,1),0)
th, img_threshold=cv2.threshold(img_filtered,190,255,cv2.THRESH_BINARY_INV)

img_open=cv2.morphologyEx(img_threshold,cv2.MORPH_OPEN,np.ones((1,1111),np.uint8))
img_close=cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))

        
th,img_threshold_1=cv2.threshold(img,190,255,cv2.THRESH_BINARY_INV)
hold_0,contours_0,hierarchy=cv2.findContours(img_threshold_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
blackhat=cv2.subtract(img_threshold_1,img_close)

bh_open = [cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))]
bh_closed = [cv2.morphologyEx(bh_open[0],cv2.MORPH_CLOSE,np.ones((1,1),np.uint8))]

for index in range(11):
    bh_open.append(cv2.morphologyEx(bh_closed[index],cv2.MORPH_OPEN,np.ones((2,3),np.uint8)))
    bh_closed.append(cv2.morphologyEx(bh_open[index+1],cv2.MORPH_CLOSE,np.ones((index+2,1),np.uint8)))
    #cv2.imshow('bh_open',bh_open[index+1])
#    if index>7:
#        cv2.imshow(str(index),bh_closed[index+1])
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
cv2.imshow('bh_closed',bh_closed[5])
cv2.waitKey(0)
cv2.destroyAllWindows()

#blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
#blackhat_close=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
#blackhat_close_1=cv2.morphologyEx(blackhat_close,cv2.MORPH_CLOSE,np.ones((4,1),np.uint8))
#blackhat_close_2=cv2.morphologyEx(blackhat_close_1,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#blackhat_close_3=cv2.morphologyEx(blackhat_close_2,cv2.MORPH_CLOSE,np.ones((6,1),np.uint8))
#blackhat_open_1=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_OPEN,np.ones((2,4),np.uint8))
#blackhat_close_4=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
#
#blackhat_close_5=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#difference=cv2.subtract(blackhat_close_5,blackhat_close_4)
#show_boxes=blackhat_close_4.copy()
#
#hold_0,contours_1,hierarchy=cv2.findContours(blackhat_close_4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours_1))
#for c in contours_1:
#    x,y,w,h=cv2.boundingRect(c)
#    show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
#        
#    #
##cv2.imshow('img_threshold',img_threshold)
##cv2.imshow('img_close',img_close)
##cv2.imshow('blackhat',blackhat)
##cv2.imshow('blackhat_close',blackhat_close)
##cv2.imshow('blackhat_close_1',blackhat_close_1)
##cv2.imshow('blackhat_close_2',blackhat_close_2)
##cv2.imshow('blackhat_close_3',blackhat_close_3)
##cv2.imshow('blackhat_close_4',blackhat_close_4)
##cv2.imshow('blackhat_close_5',blackhat_close_5)
##cv2.imshow('show_boxes',show_boxes)
#        #
##cv2.imshow('difference',difference)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
    
    
    
    
    
#    
#    
#    for index in range(15):
#    string='source/scores/img_'+str(index+25)+'.png'
#    print(string,index)
#    img=cv2.imread(string,cv2.IMREAD_GRAYSCALE)
#    height,width=img.shape[:2]
#    filter_arg=int(width*0.5)
#    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
#    img_filtered=cv2.GaussianBlur(img,(filter_arg,1),0)
#    th, img_threshold=cv2.threshold(img_filtered,190,255,cv2.THRESH_BINARY_INV)
#    
#    img_open=cv2.morphologyEx(img_threshold,cv2.MORPH_OPEN,np.ones((1,1111),np.uint8))
#    img_close=cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    
#    
#    th,img_threshold_1=cv2.threshold(img,190,255,cv2.THRESH_BINARY_INV)
#    hold_0,contours_0,hierarchy=cv2.findContours(img_threshold_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    print(len(contours_0))
#    blackhat=cv2.subtract(img_threshold_1,img_close)
#    
#    
#    blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
#    blackhat_close=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
#    blackhat_close_1=cv2.morphologyEx(blackhat_close,cv2.MORPH_CLOSE,np.ones((4,1),np.uint8))
#    blackhat_close_2=cv2.morphologyEx(blackhat_close_1,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    blackhat_close_3=cv2.morphologyEx(blackhat_close_2,cv2.MORPH_CLOSE,np.ones((6,1),np.uint8))
#    blackhat_open_1=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_OPEN,np.ones((2,4),np.uint8))
#    blackhat_close_4=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
#    
#    blackhat_close_5=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    difference=cv2.subtract(blackhat_close_5,blackhat_close_4)
#    show_boxes=blackhat_close_4.copy()
#    
#    hold_0,contours_1,hierarchy=cv2.findContours(blackhat_close_4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    print(len(contours_1))
#    for c in contours_1:
#        x,y,w,h=cv2.boundingRect(c)
#        show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
#        
#        #
#    #cv2.imshow('img_threshold',img_threshold)
#    #cv2.imshow('img_close',img_close)
#    #cv2.imshow('blackhat',blackhat)
#    #cv2.imshow('blackhat_close',blackhat_close)
#    #cv2.imshow('blackhat_close_1',blackhat_close_1)
#    #cv2.imshow('blackhat_close_2',blackhat_close_2)
#    #cv2.imshow('blackhat_close_3',blackhat_close_3)
#    #cv2.imshow('blackhat_close_4',blackhat_close_4)
#    #cv2.imshow('blackhat_close_5',blackhat_close_5)
#    cv2.imshow('show_boxes',show_boxes)
#        #
#    #cv2.imshow('difference',difference)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    
#        