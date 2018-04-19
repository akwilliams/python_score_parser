import cv2
import os
import random
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

for index in range(1):
    string='source/scores/img_'+str(index+20)+'.png'
    print(string)
    img=cv2.imread(string,cv2.IMREAD_GRAYSCALE)
    height,width=img.shape[:2]
    filter_arg=int(width*0.125)
    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
    img_filtered=cv2.GaussianBlur(img,(filter_arg,1),0)
    th, img_threshold=cv2.threshold(img_filtered,165,255,cv2.THRESH_BINARY_INV)
    
    img_open=cv2.morphologyEx(img_threshold,cv2.MORPH_OPEN,np.ones((1,75),np.uint8))
    img_close=cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
    img_closed_blr=cv2.GaussianBlur(img_close,(3,3),0)
    
    img_closed_thresh=cv2.threshold(img_closed_blr,101,255,cv2.THRESH_BINARY)[1]
    
    img_y_filter=cv2.GaussianBlur(img,(1,55),0)
    th,img_y_thresh=cv2.threshold(img_y_filter,150,255,cv2.THRESH_BINARY_INV)
    img_y_open=cv2.morphologyEx(img_y_thresh,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    img_y_close=cv2.morphologyEx(img_y_open,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    
    modded=cv2.subtract(img_closed_thresh,img_y_close)
    
    th,img_threshold_1=cv2.threshold(img,155,255,cv2.THRESH_BINARY_INV)
    hold_0,contours_0,hierarchy=cv2.findContours(img_threshold_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    blackhat=cv2.subtract(img_threshold_1,modded)

    
    blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    blackhat_closed=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    blackhat_closed_blr=cv2.GaussianBlur(blackhat_closed,(1,15),4)
        
#    cv2.imshow('test',blackhat_closed)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    show_boxes=blackhat_closed.copy()
    hold_0,contours_0,hierarchy=cv2.findContours(blackhat_closed_blr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    boxes={'x0':[],'y0':[],'x1':[],'y1':[],'xC':[],'yC':[],'width':[],'height':[],'ratio':[],'area':[],'angle':[],'pixel_mean':[],'pixel_mean_q0':[],'pixel_mean_q1':[],'pixel_mean_q2':[],'pixel_mean_q3':[]}
    for c in contours_0:
        x,y,w,h=cv2.boundingRect(c)
        show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
        boxes['x0'].append(x)
        boxes['y0'].append(y)
        boxes['x1'].append(x+w)
        boxes['y1'].append(y+h)
        boxes['xC'].append(int(x+((w+x)*0.5)))
        boxes['yC'].append(int(y+((h+y)*0.5)))
        boxes['width'].append(w)
        boxes['height'].append(h)
        boxes['ratio'].append(h/w)
        boxes['area'].append(h*w)
        if len(c)>4:
            (x,y),(MA,ma),angle=cv2.fitEllipse(c)
        else:
            angle=-1
        boxes['angle'].append(angle)
        value=show_boxes[int(y):int(y+h),int(x):int(x+w)]
        boxes['pixel_mean'].append(np.mean(value))
        if w >= 2 and h >= 2:
            y_0,y_1,x_0,x_1=int(y),int(y+(h/2)),int(x),int(x+(w/2))
            value=show_boxes[y_0:y_1,x_0:x_1]
            boxes['pixel_mean_q0'].append(np.mean(value))
            y_0,y_1,x_0,x_1=int(y+(h/2)),int(y+h),int(x),int(x+(w/2))
            value=show_boxes[y_0:y_1,x_0:x_1]
            boxes['pixel_mean_q1'].append(np.mean(value))
            y_0,y_1,x_0,x_1=int(y),int(y+(h/2)),int(x+(w/2)),int(x+w)
            value=show_boxes[y_0:y_1,x_0:x_1]
            boxes['pixel_mean_q2'].append(np.mean(value))
            y_0,y_1,x_0,x_1=int(y+(h/2)),int(y+h),int(x+(w/2)),int(x+w)
            value=show_boxes[y_0:y_1,x_0:x_1]
            boxes['pixel_mean_q3'].append(np.mean(value))
        else:
            boxes['pixel_mean_q0'].append(-1)
            boxes['pixel_mean_q1'].append(-1)
            boxes['pixel_mean_q2'].append(-1)
            boxes['pixel_mean_q3'].append(-1)
        df=pd.DataFrame(data=boxes)   

    item=' '
    index=94
    hold={'mult':[],'g_clef':[],'f_clef':[],'c_clef':[],'sharp':[],'flat':[],'natural':[],'bracket':[],'other':[],'g_clef_8':[],'f_clef_8':[],'c_clef_8':[]}

    df=df.sort_values(by=['area'],ascending=[False])
    df['numrow']=df.index.tolist()
    df=df.reset_index(drop=True)

    #Find Treble Clefs

#    #Shape and pattern parsing
#
#    df_1=df.loc[df['ratio']>2.5].copy()
#    df_1=df_1.loc[df_1['ratio']<3.1]
#    df_1=df_1.loc[df_1['area']<15*df['area'].mean()]
#    df_1=df_1.loc[df_1['area']>3500]
#    df_1=df_1.round({'area':-1,'ratio':1,'pixel_mean':0})
#    df_1=df_1.loc[df_1['ratio']<df_1.iloc[:8,:]['ratio'].mode()[0]*1.03]
#    df_1=df_1.loc[df_1['ratio']>df_1.iloc[:8,:]['ratio'].mode()[0]*0.97]
#    if len(df_1.iloc[:6,:]['width'].mode())==1:
#        df_1=df_1.loc[df_1['width']<df_1.iloc[:6,:]['width'].mode()[0]*1.02]
#        df_1=df_1.loc[df_1['width']>df_1.iloc[:6,:]['width'].mode()[0]*0.98]
## 
        
        
    df_1=df.loc[df['ratio']>2.5].copy()
    df_1=df_1.loc[df_1['ratio']<3.1]
    df_1=df_1.loc[df_1['area']<20000]
    df_1=df_1.loc[df_1['area']>3000]
    df_1=df_1.loc[df_1['pixel_mean_q0']>df_1['pixel_mean_q1']]
    df_1=df_1.loc[df_1['pixel_mean_q0']>df_1['pixel_mean_q2']]
    df_1=df_1.loc[df_1['pixel_mean_q0']>df_1['pixel_mean_q3']]
    
    df_1=df_1.loc[df_1['x0']<df_1['x0'].min()*2]
    if df_1['x0'].max()!=df_1['x0'].min():
        df_1=df_1.loc[df_1['x0']<df_1['x0'].max()]

    num=len(df_1.index.tolist())
    if num > 5:
        num=5

    for val in range(num):    
        info=df_1.iloc[val:,:]
        w,h=int(info['width'].tolist()[0]*0.25),int(info['height'].tolist()[0]*0.25)
        if w%2!=1:
            w=w-1
        if h%2!=1:
            h=h-1
        img_dup=img.copy()
        img_dup_blr=cv2.GaussianBlur(img_dup,(w,h),0)
        template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
        template_blr=cv2.GaussianBlur(template,(w,h),0)
    
        res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        cv2.imshow('template',template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        difference=802
        precision=0.105
    
        for index in range(50):
            print(difference,index)
            match_locations=np.where(res<=precision)
#            show_locations=img.copy()
            df_2=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
            difference=df_2['x'].max()-df_2['x'].min()
            if difference<=801 or len(df_2.index.tolist())<1:# or type(difference)!='numpy.int64'::
#                match_locations=np.where(res<=precision-(precision*0.1))
#                show_locations=img.copy()
#                df_2=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
                print(difference)
                break
            precision=precision-0.005
        hold=df_2.iloc[0:1,:]
        df_2['delta_x']=df_2['x'].diff().shift(1).fillna(df_2['x'].diff().shift(-1))
        df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
        df_2=df_2.loc[df_2['delta_y']>1]
        if len(df_2.index.tolist())>1:
            break
    df_2=df_2.append(hold)    
    if len(df_2.index.tolist())>0:
        w,h=info['width'].tolist()[0],info['height'].tolist()[0]
        for index,row in df_2.iterrows():
            cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+w),int(row['y']+h)),[155,155,155],2)
        
        
        df_2=df_2.sort_values(by=['y'])
        df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
       
        cv2.imshow('thing',img_dup)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
df_3=df.loc[df['x0']>=df_2['x'].min()*0.95].copy()
df_3=df_3.loc[df_3['x0']<=df_2['x'].min()*1.05]
df_3=df_3.loc[df_3['x1']<=df_1['x1'].max()*1.2]
df_3=df_3.loc[df_3['height']<=df_1['height'].max()]
df_3=df_3.loc[df_3['width']<=df_1['width'].max()*1.1]
df_3=df_3.sort_values(by=['y0'],ascending=[True])

locations=[]

for index,row in df_3.iterrows():
    for val in df_2['x'].tolist():
        if row['x0']-10<val and row['x0']>val:
            locations.append(index)

places = df_3.index.tolist()
other = 

df_3=df_3.loc[df_3.index.tolist()!=locations]


if len(df_3.index.tolist())>0:
#    w,h=info['width'].tolist()[0],info['height'].tolist()[0]
    img_dup=img.copy()
    for index,row in df_3.iterrows():
        cv2.rectangle(img_dup,(int(row['x0']),int(row['y0'])),(int(row['x1']),int(row['y1'])),[155,155,155],2)
        
    cv2.imshow('thing',img_dup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


