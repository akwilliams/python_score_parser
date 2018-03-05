import cv2
import os
import random
import numpy as np
import csv
import pandas as pd
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
    if isinstance(img,str):
        temp=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    else:
        temp=img
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
                    cv2.drawContours(temp_th,[box],0,(150,150,150),2)
                
    return used_bx,temp_th

def get_total_bx_area(boxes):
    temp=0
    for box in boxes:
        #print(cv2.minAreaRect(box)[0])
        temp=temp+cv2.contourArea(box[1])
    return temp

def write_bndng_bx_csv(calc_pass,ident):
    with open('test.csv','w',newline='') as csvfile:
        fieldnames=['id','angle','width','height','center_x','center_y']
        data_writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
        data_writer.writeheader()
        for instance in calc_pass:
            for box in instance:
                rect=cv2.minAreaRect(box[1])
                data_writer.writerow({'id':ident,'angle':box[0],'width':rect[1][1],'height':rect[1][0],'center_x':rect[0][0],'center_y':rect[0][1]})

def write_bndng_bx_pd_df(calc_pass,identity):
    temp_lib_0={'id':[],'angle':[],'center_x':[],'center_y':[]}
    temp_lib_1={'height':[],'width':[]}
    for instance in calc_pass:
        for box in instance:
            rect=cv2.minAreaRect(box[1])
            temp_lib_0['id'].append(identity)
            temp_lib_0['angle'].append(box[0])
            temp_lib_1['width'].append(rect[1][1])
            temp_lib_1['height'].append(rect[1][0])
            temp_lib_0['center_x'].append(rect[0][0])
            temp_lib_0['center_y'].append(rect[0][1])
    temp_df_0=pd.DataFrame(data=temp_lib_0)
    temp_df_1=pd.DataFrame(data=temp_lib_1)
    return temp_df_0,temp_df_1
            
            
def find_staves(img):
    stave_temp=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    height,width=stave_temp.shape[:2]
    filter_arg=int(width/2)#This needs to be dynamid
    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
    hold,img=draw_boxes_by_params(stave_temp,filter_arg,1,0,165,[-1,-1],[-1,-1],True)#threshold needs to be dynamic
    df_0,df_1 = write_bndng_bx_pd_df([hold],'stave_line')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df = pd.concat([df_0,df_1],axis=1)
    for i in range(df.shape[0]%5):    
        mean,std=df['width'].mean(axis=0),df['width'].std(axis=0)
        df['id']=np.where(np.abs(df['width']-mean)>std,'miss_id,find_staves','stave_line')
        df=df.drop(df.index[np.where(df['id']!='stave_line')[0][0]])
    print(df)
    if df.shape[0]%5!=0:
        print('something is off')
        #Need to parse the data to find the issue
    return df,img


data,img_0=find_staves('score.png')
cv2.imshow('img',img_0)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(type('thing'))
boxes=[]
#run_0,img=draw_boxes_by_params('score.png',11,11,9,150,[-1,-1],[-0.005,0.005],True)
#run_1,img=draw_boxes_by_params('score.png',17,17,9,150,[2500,10000],[0.0005,0.33],True)
num = 884/21
num = num if num%2 == 1 else num+1
num=int(num)
print(num)
img_0=cv2.imread('score.png',cv2.IMREAD_GRAYSCALE)
height,width=img_0.shape[:2]
print(width)
run_1,img=draw_boxes_by_params('source/scores/img_4.png',1,1,0,150,[-1,-1],[-1,-1],True)
#img_2=cv2.imread('score.png',cv2.IMREAD_GRAYSCALE)
#img_2=cv2.GaussianBlur(img_2,(3,3),9)
#th,img_2=cv2.threshold(img_2,45,255,cv2.THRESH_BINARY_INV)
#img_2=cv2.GaussianBlur(img_2,(9,9),1)
#img_3=cv2.subtract(img_2,img)
#img_3=cv2.imread('score.png',cv2.IMREAD_GRAYSCALE)
#th,img_4=cv2.threshold(img_3,45,255,cv2.THRESH_BINARY_INV)
#img_4=cv2.GaussianBlur(img_3,(1257,1),0)
#th,img_4=cv2.threshold(img_3,150,255,cv2.THRESH_BINARY)
#total_area=get_total_bx_area(boxes)
#boxes.append(run_0)
#boxes.append(run_1)
#write_bndng_bx_csv(boxes)
#print(boxes)


#load_srcs()





cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()