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
            
            
def parse_staves(img,thresh=100,width_ratio=(1/2)):
    stave_temp=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    height,width=stave_temp.shape[:2]
    filter_arg=int(width*width_ratio)
    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
    hold,img=draw_boxes_by_params(stave_temp,filter_arg,1,0,thresh,[-1,-1],[-1,-1],False)
    df_0,df_1 = write_bndng_bx_pd_df([hold],'stave_line')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df = pd.concat([df_0,df_1],axis=1)
    return df,img

def find_score_bounds(img,thresh=0.5):
    hold,score_temp=draw_boxes_by_params(img,1,1,0,200,[-1,-1],[-1,-1],False)
    height, width=score_temp.shape[:2]
    temp_df=pd.DataFrame([[0,width/2,height/2,'img_bounds',height,width,(height*width)]],columns=['angle','center_x','center_y','id','height','width','area'])
    df,df_1=write_bndng_bx_pd_df([hold],'score_bounds')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df_1['area']=df_1.apply(lambda row: row.height*row.width, axis=1)
    df=pd.concat([df,df_1],axis=1)
    df=df.sort_values('area',ascending=False)
    df=df.reset_index(drop=True)
    maximum=df['area'][0]
    df=df.drop(df.index[np.where((df['area']/maximum)<thresh)[0]])
    df=df.append(temp_df)
    df=df.reset_index(drop=True)
    return df


def find_stave_data(img,init_thresh=45,init_w_ratio=(1/2)):
    #find the bounds of the total image
    
    temp_df,temp_img=parse_staves(img,init_thresh,init_w_ratio)
    #Now I need to analyse the data
    '''
    Kind of things I need to look at:
    What if it is less than 5:
            if less than 5 raise the threshold and re-parse
            
    What if it is more than 5 but not a multiple of five
    What if there are 5 or multipe of 5 staves, but they are not stave lines but something else 
    
    '''






#    for i in range(df.shape[0]%5):  
#        print(df['width'].std(axis=0))
#        mean,std=df['width'].mean(axis=0),df['width'].std(axis=0)
#        df['id']=np.where(np.abs(df['width']-mean)>std,'miss_id,find_staves','stave_line')
#        df=df.drop(df.index[np.where(df['id']!='stave_line')[0][0]])
#        print(df['width'].std(axis=0))














data,img_0=parse_staves('source/scores/img_4.png',190,(2/3))
img_1=cv2.imread('score.png')

df=find_score_bounds('score.png',0.65)
print(df)


#
#hold,img=draw_boxes_by_params('source/scores/img_6.png',1,1,0,200,[-1,-1],[-1,-1],True)
#height, width=img.shape[:2]
#temp_df=pd.DataFrame([[0,width/2,height/2,'img_bounds',height,width,(height*width)]],columns=['angle','center_x','center_y','id','height','width','area'])
#df,df_1=write_bndng_bx_pd_df([hold],'score_bounds')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_1['area']=df_1.apply(lambda row: row.height*row.width, axis=1)
#df=pd.concat([df,df_1],axis=1)
#df=df.sort_values('area',ascending=False)
#df=df.reset_index(drop=True)
#print(df.loc[2])
#maximum=df['area'][0]
#print(maximum)
#print(np.where((df['area']/maximum)<0.5)[0])
#df=df.drop(df.index[np.where((df['area']/maximum)<0.5)[0]])
#print(df)
#df=df.append(temp_df)
#df=df.reset_index(drop=True)
#print(df)

#
#df_1['area']=df_1.apply(lambda row: row.height*row.width, axis=1)
#mean,std=df_1['area'].mean(axis=0),df_1['area'].std(axis=0)
#df_1['area']=np.where(df_1['area']<mean,-1,df_1['area'])
#df_1=df_1.drop(df_1.index[np.where(df_1['area']==-1.0)[0]])
#print(df_1)
#mean,std=df_1['area'].mean(axis=0),df_1['area'].std(axis=0)
#print(df_1['area'].max(axis=0)-df_1['area'].min(axis=0))
#print(mean,std)

cv2.imshow('img',img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
