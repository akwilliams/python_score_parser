import cv2
import os
import random
import numpy as np
import csv
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
#Loads a score with presets for cv functions          
def draw_boxes_by_params(img,gBlur_x=11,gBlur_y=11,gBlur_std_dev=9,thresh=125,bx_area_thresh=[-1,-1],bx_angle_thresh=[-1,-1],draw_rects=False):
    temp=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    temp_blr=cv2.GaussianBlur(temp,(gBlur_x,gBlur_y),gBlur_std_dev)
    th,temp_th=cv2.threshold(temp_blr,thresh,255,cv2.THRESH_BINARY_INV)
    temp2,contours,hierarchy=cv2.findContours(temp_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    used_bx=[]       
    for c in contours:
        area=cv2.contourArea(c)
        if area >= bx_area_thresh[0] and area <= bx_area_thresh[1] or bx_area_thresh[0]<=-1 or bx_area_thresh[1]<=-1:
            rect=cv2.minAreaRect(c)
            angle=(-1*rect[2]) if rect[1][0]>rect[1][1] else (-1*rect[2]-90)
            angle=angle if angle>=0 else angle+360
            angle=np.radians(angle)/np.pi if np.radians(angle)/np.pi<=0.5 else np.radians(angle)/np.pi-2
            if angle >= bx_angle_thresh[0] and angle <= bx_angle_thresh[1] or bx_angle_thresh[0]<-0.5 or bx_angle_thresh[1]>0.5:
                used_bx.append([])
                used_bx[len(used_bx)-1].append(angle)
                used_bx[len(used_bx)-1].append(c)
                if draw_rects == True:
                    box=cv2.boxPoints(rect)
                    box=np.int0(box)
                    cv2.drawContours(temp,[box],0,(150,150,150),2)
                
    return used_bx,temp

def get_total_bx_area(boxes):
    temp=0
    for box in boxes:
        #print(cv2.minAreaRect(box)[0])
        temp=temp+cv2.contourArea(box[1])
    return temp

def write_bndng_bx_csv(calc_pass):
    with open('test.csv','w',newline='') as csvfile:
        fieldnames=['angle','len_wdth_rat','center_x','center_y']
        data_writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
        data_writer.writeheader()
        for instance in calc_pass:
            for box in instance:
                rect=cv2.minAreaRect(box[1])
                data_writer.writerow({'angle':box[0],'len_wdth_rat':rect[1][0]/rect[1][1],'center_x':rect[0][0],'center_y':rect[0][1]})
            

boxes=[]
run_0,img=draw_boxes_by_params('score.png',11,11,9,150,[-1,-1],[-0.005,0.005],False)
run_1,img=draw_boxes_by_params('score.png',17,17,9,150,[-1,-1],[0.0005,0.33],True)
#total_area=get_total_bx_area(boxes)
boxes.append(run_0)
boxes.append(run_1)
write_bndng_bx_csv(boxes)
#print(boxes)


#load_srcs()


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()