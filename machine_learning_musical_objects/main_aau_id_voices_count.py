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
            temp_lib_1['width'].append(int(rect[1][1]))
            temp_lib_1['height'].append(int(rect[1][0]))
            temp_lib_0['center_x'].append(int(rect[0][0]))
            temp_lib_0['center_y'].append(int(rect[0][1]))
    temp_df_0=pd.DataFrame(data=temp_lib_0)
    temp_df_1=pd.DataFrame(data=temp_lib_1)
    return temp_df_0,temp_df_1
            
            
def parse_staves(img,thresh=100,width_ratio=(1/2)):
    if isinstance(img,str):
        img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
#    else:
#        temp=img
    stave_temp=img
    height,width=stave_temp.shape[:2]
    filter_arg=int(width*width_ratio)
    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
    hold,img=draw_boxes_by_params(stave_temp,filter_arg,1,0,thresh,[-1,-1],[-1,-1],False)
    df_0,df_1 = write_bndng_bx_pd_df([hold],'stave_line')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df = pd.concat([df_0,df_1],axis=1)
    return df,img

def find_score_bounds(img,thresh=0.5,size_ratio=0.1):
    temp_img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    height,width=temp_img.shape[:2]
    hold,score_temp=draw_boxes_by_params(temp_img,1,1,0,200,[(height*width*size_ratio),(height*width)],[-1,-1],True)
    temp_df=pd.DataFrame([[0,width/2,height/2,'img_bounds',height,width,(height*width)]],columns=['angle','center_x','center_y','id','height','width','area'])
    df,df_1=write_bndng_bx_pd_df([hold],'score_bounds')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df_1['area']=df_1.apply(lambda row: row.height*row.width, axis=1)
    df=pd.concat([df,df_1],axis=1)
    df=df.drop(df.index[np.where((df['area']/df['area'].max()<thresh))[0]])
    df=df.append(temp_df)
    df=df.reset_index(drop=True)
    return df,score_temp

def calc_perc_off_mean(df,index,destination):
    mean=df[index].mean()
    df[destination]=np.abs((df[index]-mean)/mean)
    return df

def drop_by_perc_off_mean(df,ident,destination,threshold):
    df=calc_perc_off_mean(df,ident,destination)
    while df[destination].max()>threshold:
        df=df.drop(df[destination].idxmax())
        df=calc_perc_off_mean(df,ident,destination)
    return df

def drop_value_below_threshold(df,ident,threshold):
    print(df[ident])
    df[ident].abs()
    print(df[ident])
    df=df[df[ident]>threshold]
    return df

def find_stave_data(img,bounds_thresh=0.5,bounds_size_ratio=0.01,init_filter_thresh=45,width_ratio_std=0.8,width_thresh=0.2,delta_y_thresh=0.4):

    img_info,temp_img=find_score_bounds(img,bounds_thresh,bounds_size_ratio)
    width_ratio=width_ratio_std*img_info['width'][np.where(img_info['id']=='score_bounds')[0][0]]/img_info['width'][np.where(img_info['id']=='img_bounds')[0][0]]
    df,temp_img_0=parse_staves(img,init_filter_thresh,width_ratio)
    df=drop_by_perc_off_mean(df.copy(),'width','percent_off_mean_width',width_thresh)   
    df['delta_y']=df.center_y.diff().shift(-1).fillna(0)
    df_delta_y=df.groupby(['delta_y']).size().reset_index(name='count')
    mean=df_delta_y['delta_y'].mean()-df_delta_y['delta_y'].mode()
    df_delta_y['over_mean']=df_delta_y.loc[df_delta_y['delta_y']<mean,['count']].sum(axis=1)
    df_delta_y['under_mean']=df_delta_y.loc[df_delta_y['delta_y']>mean,['count']].sum(axis=1)
    systems=len(img_info.index[np.where(img_info['id']=='score_bounds')])
    score={'staves_in_system':np.around(((df_delta_y['over_mean'].sum(axis=0)-(systems-1))/systems)+1),'system_count':systems}
    #print(df['center_y'].min(),df['center_y'].min()-(4*df_delta_y['delta_y'][np.where(df_delta_y['count']==df_delta_y['count'].max())[0][0]]),df,df_delta_y)
    find_index=df_delta_y.loc[df_delta_y['count'].idxmax()]['delta_y']*4
    find_x_val=df.center_y[df.center_y==390].index.tolist()[0]
    print(df['center_y'][find_x_val])
    print(find_x_val)
    print(find_index)
    stave_0=[[df['center_x'][find_x_val]-(df['width'][find_x_val]/2),df['center_y'].min()],[df['center_x'][find_x_val]+(df['width'][find_x_val]/2),df['center_y'].min()-(4*df_delta_y['delta_y'][np.where(df_delta_y['count']==df_delta_y['count'].max())[0][0]])]]
    print(stave_0)
    #score_bounds={}
    return temp_img_0


img_2=find_stave_data('source/scores/img_9.png',init_filter_thresh=145,width_thresh=0.2,delta_y_thresh=0.2)

cv2.imshow('img',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
