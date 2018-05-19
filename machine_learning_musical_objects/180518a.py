import cv2
import numpy as np
import pandas as pd
from scipy import stats
import os
from PIL import Image

'''Import Tesseract'''
import pytesseract 
#For MacOS
pytesseract.pytesseract.tesseract_cmd ='/usr/local/bin/tesseract'
##For WindowsOS
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'



##img=Image.open('source/scores/bartok_piano/img_1.png')
#img=cv2.imread('source/scores/beethoven_op81/img_0.png',cv2.IMREAD_GRAYSCALE)
#test_img=Image.fromarray(img)
#text=pytesseract.image_to_string(test_img,lang='eng')
#print(text)
#
#img=Image.open('source/scores/beethoven_op81/img_0.png')
#text=pytesseract.image_to_string(img,lang='deu')
#print(text)
#
#img=Image.open('source/scores/strauss_wanders_strumlied/img_1.png')
#text=pytesseract.image_to_string(img,lang='eng')
#print(text)
#df_parsing=pd.DataFrame(data={'content':text.split('\n')})
#df_parsing=df_parsing.loc[df_parsing['content'].str.len()>1]
#



def init_img_filter(img):
    img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
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
    
    show_boxes=blackhat_closed.copy()
    hold_0,contours_0,hierarchy=cv2.findContours(blackhat_closed_blr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    boxes={'x0':np.array([],dtype=np.uint16),'y0':np.array([],dtype=np.uint16),'x1':np.array([],dtype=np.uint16),'y1':np.array([],dtype=np.uint16),'xC':np.array([],dtype=np.uint16),'yC':np.array([],dtype=np.uint16),'width':np.array([],dtype=np.uint16),'height':np.array([],dtype=np.uint16),'ratio':np.array([],dtype=np.float32),'area':np.array([],dtype=np.float32),'angle':np.array([],dtype=np.float32),'pixel_mean':np.array([],dtype=np.uint16),'pixel_mean_q0':np.array([],dtype=np.uint16),'pixel_mean_q1':np.array([],dtype=np.uint16),'pixel_mean_q2':np.array([],dtype=np.uint16),'pixel_mean_q3':np.array([],dtype=np.uint16)}
    for c in contours_0:
        x,y,w,h=cv2.boundingRect(c)
        show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
        boxes['x0']=np.append(boxes['x0'],[x])
        boxes['y0']=np.append(boxes['y0'],[y])
        boxes['x1']=np.append(boxes['x1'],[x+w])
        boxes['y1']=np.append(boxes['y1'],[y+h])
        boxes['xC']=np.append(boxes['xC'],[int(x+((w+x)*0.5))])
        boxes['yC']=np.append(boxes['yC'],[int(y+((h+y)*0.5))])
        boxes['width']=np.append(boxes['width'],[w])
        boxes['height']=np.append(boxes['height'],[h])
        boxes['ratio']=np.append(boxes['ratio'],[h/w])
        boxes['area']=np.append(boxes['area'],[h*w])
        if len(c)>4:
            (x,y),(MA,ma),angle=cv2.fitEllipse(c)
        else:
            angle=-1
        boxes['angle']=np.append(boxes['angle'],[angle])
        boxes['pixel_mean']=np.append(boxes['pixel_mean'],np.mean(show_boxes[int(y):int(y+h),int(x):int(x+w)]))
        if w >= 2 and h >= 2:
            for index_0 in range(4):
                if index_0%2==0:
                    x_0=int(x)
                    x_1=int(x+((w+x)/2))
                else:
                    x_0=int(x+(w/2))
                    x_1=int(x+w)
                if index_0<2:
                    y_0=int(y)
                    y_1=int(y+(h/2))
                else:
                    y_0=int(y+(h/2))
                    y_1=int(y+h)
                    
                value=show_boxes[y_0:y_1,x_0:x_1]
                string='pixel_mean_q'+str(index_0)
#                print(string,x_0,x_1,y_0,y_1,value)
#                print(np.mean(value))
                if np.mean(value)!=np.nan:
                    boxes[string]=np.append(boxes[string],np.mean(value))
                else:
                    boxes[string]=np.append(boxes[string],int(0))
        else:
            boxes['pixel_mean_q0']=np.append(boxes['pixel_mean_q0'],[0])
            boxes['pixel_mean_q1']=np.append(boxes['pixel_mean_q1'],[0])
            boxes['pixel_mean_q2']=np.append(boxes['pixel_mean_q2'],[0])
            boxes['pixel_mean_q3']=np.append(boxes['pixel_mean_q3'],[0])

    df=pd.DataFrame(data=boxes)
    return df,img


def find_g_clefs(df,img):
    df_0=df.loc[df['x0']<img.shape[:2][1]/6]
    df_0=df_0.loc[df_0['ratio']>2.5]
    df_0=df_0.loc[df_0['ratio']<3.1]
    df_0=df_0.loc[df_0['area']<25000]
    df_0=df_0.loc[df_0['area']>3000]
    
    df_0=df_0.sort_values(by=['area','x0'],ascending=[False,True])
    df_0['x0']=df_0['x0'].round(-1)
    container_mins=np.array([],dtype=np.float32)
    for index_0 in range(df_0['x0'].mode().shape[0]):
        container_mins=np.append(container_mins,[df_0.loc[df_0['x0']==df_0['x0'].mode()[index_0]]['x0'].min()])
    df_0=df_0.loc[df_0['x0']==df_0['x0'].mode()[np.argmin(container_mins)]]

    w,h=int(df_0['width'].values[0]*0.25),int(df_0['height'].values[0]*0.25)
    if w%2!=1:
        w=w-1
    if h%2!=1:
        h=h-1
        
    img_dup=img.copy()
    img_dup_blr=cv2.GaussianBlur(img_dup,(w,h),0)
    template=img[df_0['y0'].values[0]:df_0['y1'].values[0],df_0['x0'].values[0]:df_0['x1'].values[0]]
    template_blr=cv2.GaussianBlur(template,(w,h),0)
    
#    cv2.imshow('template',template)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    match_locations=np.where(res<=0.035)
    df_1=pd.DataFrame(data={'x':np.array(match_locations[1],dtype=np.uint16),'y':np.array(match_locations[0],dtype=np.uint16)})
    
    df_1=df_1.sort_values(by=['y'],ascending=[True])
    
    hold=df_1.iloc[:1,:].copy()
    df_1['delta_y']=df_1['y'].diff().shift(1).fillna(df_1['y'].diff().shift(-1))
    df_1=df_1.loc[df_1['delta_y']>1]
    df_1=df_1.append(hold)
    df_1=df_1.sort_values(by=['y'],ascending=[True])
    
#    if df_1.shape[0]>0:
#        w,h=df_0['width'].values[0],df_0['height'].values[0]
#        img_dup=img.copy()
#        for index,row in df_1.iterrows():
#            cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+w),int(row['y']+h)),[155,155,155],2)
#        
#        cv2.imshow('thing',img_dup)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
      
    length=df_1.shape[0]
    w,h=df_0['width'].values[0],df_0['height'].values[0]
    df_1['kind']=['g_clef']*length
    df_1['width']=[w]*length
    df_1['height']=[h]*length
    df_1['area']=df_1['width'].multiply(df_1['height'])
    
    return df_1

def find_clefs_from_reference(df_0,df_1,img):

    temp_img=img.copy()
    height,width=temp_img.shape[:2]
    temp_blr=cv2.GaussianBlur(temp_img,(1,1),100,245)
    waste_0,temp_th=cv2.threshold(temp_blr,245,255,cv2.THRESH_BINARY_INV)
    
    waste_1,contours,hierarchy=cv2.findContours(temp_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    container_bounds=np.array([])
    
    for c in contours:
        area=cv2.contourArea(c)
        if area>=(height*width*0.0125) and area<=(height*width):
            x,y,w,h=cv2.boundingRect(c)
            container_bounds=np.append(container_bounds,[y,y+h])
            cv2.rectangle(temp_th,(x,y),((x+w),(y+h)),[155,155,155],2)
            
    container_change=[]
    for index in range(df_1.shape[0]):
        val_0,val_1=df_1.iloc[index:(index+2),:]['x'].max(),df_1.iloc[index:(index+2),:]['x'].min()
        val_2=val_0-val_1
        if val_2>75:
            container_change.append(index)
    for index_0 in range(len(container_change)+1):
        if index_0==(len(container_change)) and index_0>0:
            x0,x1,y0,y1=df_1.iloc[container_change[index_0-1]+1:,:]['x'].tolist()[0]*0.98,df_1.iloc[container_change[index_0-1]+1:,:]['x'].tolist()[0]*1.02,df_1.iloc[container_change[index_0-1]+1:,:]['y'].tolist()[0]*0.98,container_bounds.max()
        elif len(container_change)>0:
            if index_0==0:
                y0,y1=container_bounds.min(),df_1.iloc[(container_change[index_0]+1):,:]['y'].tolist()[0]*1.02
            else:
                y1,y0=df_1.iloc[(container_change[index_0]+1):,:]['y'].tolist()[0]*1.02,df_1.iloc[(container_change[index_0]):,:]['y'].tolist()[0]*0.98
            x0,x1=df_1.iloc[container_change[index_0]:,:]['x'].tolist()[0]*0.98,df_1.iloc[container_change[index_0]:,:]['x'].tolist()[0]*1.1
        else:
            x0,x1,y0,y1=df_1['x'].tolist()[0]*0.98,df_1['x'].tolist()[0]*1.02,0,-1
        df_2=df_0.loc[df_0['x0']>x0].copy()
        df_2=df_2.loc[df_2['x0']<x1]
        df_2=df_2.loc[df_2['y0']>y0]
        if y1>0:
            df_2=df_2.loc[df_2['y0']<y1]
        df_2=df_2.loc[df_2['x0']<((df_1['x'].max()+df_1['width'].max())*1.1)].copy()
        df_2=df_2.loc[df_2['x0']>df_1['x'].min()*0.9]
        df_2=df_2.loc[df_2['area']>0.05*df_1['area'].min()]
        df_2=df_2.sort_values(by=['y0','area'],ascending=[True,False])
        df_2['crossover']=df_2['y1'].shift().fillna(-1)
        df_2['crossover']=df_2['crossover'].subtract(df_2['y0'])
        df_2['co_scaler']=df_2['height'].shift().fillna(1)
        df_2['crossover']=df_2['crossover'].divide(df_2['co_scaler'])
        df_2['numrow']=df_2.index.tolist()
        if index_0==0:
            df_3=df_2.copy()
        else:
            df_3=df_3.append(df_2.copy())
    df_3=df_3.sort_values(by=['area'],ascending=[False])
    
    df_3=df_3.loc[df_3['width']<df_1['width'].max()*1.1]
    df_3=df_3.loc[df_3['height']<df_1['height'].max()*1.33]
    df_3=df_3.loc[df_3['height']>df_1['height'].min()*0.133]
    df_3=df_3.round({'area':-1,'height':-1,'width':0})
    df_3['delta_area']=df_3['area'].diff().shift().fillna(0)
    df_3['delta_height']=df_3['height'].diff().shift().fillna(0)
    df_3['delta_width']=df_3['width'].diff().shift().fillna(0)
    hold=df_3.iloc[0:1,:].copy()

    df_3['diff_total']=df_3['delta_area'].add(df_3['delta_height'].add(df_3['delta_width']))
    df_3=df_3.loc[df_3['diff_total'].abs()>66]
    df_3=df_3.append(hold)
    
    container_df=[]
    container_template=[]
    for index_1 in range(df_3.shape[0]):
        info=df_3.iloc[index_1:,:]
        container_template.append(df_3.iloc[index_1:(index_1+1),:].copy())
        w,h=int(info['width'].tolist()[0]*0.125),int(info['height'].tolist()[0]*0.125)
        if w%2!=1:
            w=w-1
            if w<=0:
                w=1
        if h%2!=1:
            h=h-1
            if h<=0:
                h=1
    
        img_dup=img.copy()
        img_dup_blr=cv2.GaussianBlur(img_dup,(w,h),0)
        template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
        template_blr=cv2.GaussianBlur(template,(w,h),0)

#        cv2.imshow('template',template)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

        res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        resolution=0.065
 
        match_locations=np.where(res<=resolution)
        df_4=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
        df_5=df_4.copy()
        df_5['delta_x']=df_5['x'].diff().shift().fillna(df_5['x'].diff().shift(-1))
        df_5['delta_y']=df_5['y'].diff().shift().fillna(df_5['y'].diff().shift(-1))
        df_5=df_5.loc[df_5['delta_y']>1]
        df_5=df_5.append(df_4.iloc[0:1,:].copy())
        df_5['height']=[int(info['height'].tolist()[0])]*df_5.shape[0]
        df_5['width']=[int(info['width'].tolist()[0])]*df_5.shape[0]
        container_df.append(df_5.copy())
        
      
#    print(container_df)
    for index_0 in range(len(container_df)):
        if index_0==0:
            df_6=df_1.copy()
            count=df_6.shape[0]
            df_6['pass_count']=[-1]*(count)
            df_6['y1']=df_6['y'].add(df_6['height'])
            
        df_7=container_df[index_0].copy()            
        count=df_7.shape[0]
        df_7['pass_count']=[index_0]*(count)
        df_7['y1']=df_7['y'].add(df_7['height'])
        df_7=df_7.append(df_6.copy())
        df_7=df_7.sort_values(by=['y','pass_count'],ascending=[True,False])

        df_7['delta_y']=df_7['y'].diff().shift(-1).fillna(155)
        if df_7['delta_y'].min()>108:
            df_6=df_6.append(container_df[index_0].copy())
            df_6['pass_count']=df_6['pass_count'].fillna(index_0)
    df_6=df_6.sort_values(by=['y'],ascending=[True])
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(df_6['y'].diff().shift())
    df_6=df_6.loc[df_6['y']<container_bounds.max()+100]
    df_6=df_6.loc[df_6['y']>container_bounds.min()-100]
    
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(0)
    minimum,mean,maximum=df_6.iloc[:-1,:]['delta_y'].min(),df_6.iloc[:-1,:]['delta_y'].mean(),df_6.iloc[:-1,:]['delta_y'].max()
    df_6=df_6.reset_index(drop=True)
#    print(df_6)
    check_for_more=False
    for index,row in df_6.iterrows():
        if row['delta_y']>2.3*row['height']:
            check_for_more=True
            break
    if check_for_more==True:
        for index_0 in range(5):
            if check_for_more==True:
                location=df_6['delta_y'].idxmax(axis=1)
                df_8=df_6.iloc[location:location+2].copy()
                df_9=df_0.loc[df_0['y0']>df_8['y'].min()+(minimum*0.1)].copy()
                df_9=df_9.loc[df_9['y1']<df_8['y'].max()-(minimum*0.1)]
                df_9=df_9.loc[df_9['x0']>df_8['x'].min()-20]
                df_9=df_9.loc[df_9['x0']<df_8['x'].max()+20]
                df_9=df_9.loc[df_9['area']>100]
                df_9=df_9.loc[df_9['ratio']<3.7]
                df_9=df_9.sort_values(by=['area'],ascending=[False])
                df_9=df_9.reset_index(drop=True)
                for index_1 in range(df_9.shape[0]):
                    info=df_9.iloc[index_1:,:]
                    img_dup=img.copy()
                    img_dup_blr=cv2.GaussianBlur(img_dup,(w,h),0)
                    template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
                    template_blr=cv2.GaussianBlur(template,(w,h),0)
                    
                    
#                    cv2.imshow('template',template)
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
                    
                    res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
                    resolution=0.065
         
                    match_locations=np.where(res<=resolution)
                    df_10=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
                    if df_10['x'].max()-df_10['x'].min() > 750:
                        break
                    df_10['delta_y']=df_10['y'].diff().shift(-1).fillna(2)
                    df_10=df_10.loc[df_10['delta_y']>1]
                    df_10['height']=info['y1'].tolist()[0]-info['y0'].tolist()[0]
                    df_10['width']=info['x1'].tolist()[0]-info['x0'].tolist()[0]
                    df_10['kind']='fill_missing_'+str(index_0)+'_'+str(index_1)
                    df_10['pass_count']=99
                    df_11=df_6.copy()
                    df_11=df_11.append(df_10)
                    df_11=df_11.sort_values(by=['y'],ascending=[True])
                    df_11['delta_y']=df_11['y'].diff().shift(-1).fillna(6)
                    if df_11['delta_y'].min()<5:
                        df_12=df_11.loc[df_11['delta_y']<5].copy()
                        df_12=df_12.loc[df_12['delta_y']!=0]
#                        df_12=df_11.loc[df_11['pass_count']==df_12['pass_count'].mode()[0]].copy()
                        df_13=df_10.copy()
                        for index_2 in range(df_13.shape[0]):
                            df_14=df_12.copy()
                            df_14=df_14.append(df_13.iloc[index_2:index_2+1,:].copy())
                            df_14=df_14.sort_values(by=['y'],ascending=True)
                            df_14=df_14.reset_index(drop=True)
                            df_14['delta_y']=df_14['y'].diff().shift(-1).fillna(36)
#                            print(df_14)
                            if df_14['delta_y'].min()>35:
                                df_12=df_12.append(df_13.iloc[index_2:index_2+1,:])
                                df_12=df_12.sort_values(by=['y'],ascending=[True])
                                df_12=df_12.reset_index(drop=True)
                                if df_12['pass_count'].mode().shape[0]>0:
                                    df_12['pass_count']=[df_12['pass_count'].mode()[0]]*df_12.shape[0]                  
                        string='fill_missing_'+str(index_0)+'_'+str(index_1)
                        df_6=df_6.append(df_12.loc[df_12['kind']==string])
                        
                    else:
                        df_6=df_6.append(df_10)
                        
                    df_6=df_6.sort_values(by=['y'],ascending=[True])
                    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(0)
                    check_for_more=False
                    for index,row in df_6.iterrows():
                        if row['delta_y']>2.3*row['height'] and row['delta_y']>6:
                            check_for_more=True
                            break
                    df_6=df_6.reset_index(drop=True)
                    break
            else:
                break
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(1)              
    df_6['y1']=df_6['y'].add(df_6['height'])
    df_6['delta_y1']=df_6['y'].subtract(df_6['y1'].shift()).fillna(1)
    df_6=df_6.loc[df_6['delta_y1']>-2]
    df_6=df_6.loc[df_6['delta_y1']<750]
#    print(df_6)
    #Here is where I need to parse df_6 to id C and F clefs
    container_system_info=[]
    system_count=container_bounds.shape[0]/2
    for index_0 in range(int(system_count)):
        y_bounds=container_bounds[index_0*2:(index_0*2)+2]
        df_15=df_6.loc[df_6['y']>y_bounds.min()-25].copy()
        df_15=df_15.loc[df_15['y']<y_bounds.max()+25]
        container_system_info.append(df_15.copy())
        
    return container_system_info

def calc_staff_font_info(df_0,img):
    needs_postprocessing=False
    for index_0 in range(len(df_0)):
        height,width=int(df_0[index_0]['height'].max()),int(df_0[index_0]['width'].max())
        container_font_info={'pixel_mean':[],'delta_line':[],'pass_count':[],'kind':[]}
        for index_1 in range(df_0[index_0].shape[0]):
            info=df_0[index_0].iloc[index_1:,:].copy()
            template=img[info['y'].values[0]:info['y'].values[0]+height,info['x'].values[0]:info['x'].values[0]+width]
            if width%2!=1:
                width=width-1
            if height%2!=1:
                height=height-1
            template_blr=cv2.GaussianBlur(template,(width,1),0)
            th,template_th=cv2.threshold(template_blr,int((np.mean(template_blr))*0.75),255,cv2.THRESH_BINARY_INV)

            for index_2 in range(2):
                template_open=cv2.morphologyEx(template_th,cv2.MORPH_OPEN,np.ones((1,75),np.uint8))
                template_close=cv2.morphologyEx(template_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
                template_th=template_close
                           
#            cv2.imshow('template',template_th)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
#            cv2.imshow('template_open',template_open)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            
#            cv2.imshow('template_closed',template_close)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
            df_1=pd.DataFrame(data={'row_0':template_close[:,0].copy(),'row_1':template_close[:,-1].copy()})
            df_1=df_1.divide(2)
            df_1['sum']=df_1['row_0'].add(df_1['row_1'])
            df_1=df_1.loc[df_1['sum']>200]
            df_1['numrow']=df_1.index.tolist()
            df_1['delta_p']=df_1['numrow'].diff().shift(-1).fillna(2)
#            print(df_1)
            df_1=df_1.loc[df_1['delta_p']>5]
            df_1=df_1.reset_index(drop=True)
            for index_2 in range(df_1.shape[0]):
                if df_1['delta_p'].min()/df_1['delta_p'].mean() < 0.66:
                    hold_value,hold_index=df_1['delta_p'].min(),df_1['delta_p'].idxmin()
                    if hold_index != 0 and hold_index != df_1.index.tolist()[-1]:
                        val_0,val_1=df_1['delta_p'].values[(hold_index-1)],df_1['delta_p'].values[(hold_index+1)]
                        if val_0 < val_1:
                            hold_index=hold_index-1
                        else:
                            hold_index=hold_index+1
                        
                    elif hold_index==0:
                        hold_index=1
                    elif hold_index==df_1.index.tolist()[-1]:
                        hold_index=hold_index-1
                    
                    if np.abs(hold_value+df_1['delta_p'].values[hold_index]-df_1['delta_p'].max()) < 6:
                        df_1.loc[df_1['numrow']==df_1['numrow'].values[hold_index],'delta_p']=df_1['delta_p'].values[hold_index]+hold_value
                        
                        
                    df_1=df_1.loc[df_1['delta_p']>df_1['delta_p'].min()]
                    df_1=df_1.reset_index(drop=True)
                else:
                    break
            df_1['delta_line']=df_1['delta_p'].tolist()
            template_a=img[info['y'].values[0]:int(info['y'].values[0]+template.shape[0]),info['x'].values[0]:int(info['x'].values[0]+info['width'].values[0])]
            template_blr_a=cv2.GaussianBlur(template_a,(9,1),0)
            th,template_th_a=cv2.threshold(template_blr_a,int((np.min(template_a)+(255-np.mean(template_blr_a)))*1.3),255,cv2.THRESH_BINARY_INV)
            container_font_info['pixel_mean'].append(np.mean(template_th_a))
            
            if np.abs(df_1['delta_line'].mean()-df_1['delta_line'].mode()[0]) > 2.5:
                container_font_info['delta_line'].append(df_1['delta_line'].tolist())
                needs_postprocessing=True
            else:
                container_font_info['delta_line'].append(int(df_1['delta_line'].mean()))
            
            
        df_0[index_0]['delta_line']=container_font_info['delta_line']
        df_0[index_0]['pixel_mean']=container_font_info['pixel_mean']
    
    container_postprocessing_index=[]    
    if needs_postprocessing==True:
        for index_0 in range(len(df_0)):
            df_2=df_0[index_0].copy()
            df_2=df_2.reset_index(drop=True)
            for index,row in df_2.iterrows():
                if type(row['delta_line'])==list:
                    container_postprocessing_index.append([row['pass_count'],index,index_0,[]])
    
    for data in container_postprocessing_index:
        df_3=df_0[data[2]].loc[df_0[data[2]]['pass_count']==data[0]].copy()
        for df_temp in df_0:
            df_4=df_temp.loc[df_temp['pass_count']==data[0]].copy()
            for val in df_4['delta_line'].values:
                if type(val) != list:
                    data[3].append(val)
        if len(data[3])>0:
            df_0[data[2]]['delta_line'].values[data[1]]=np.mean(data[3])
        else:
            df_0[data[2]]['delta_line'].values[data[1]]=-1
    mean=0
    for index_0 in range(len(df_0)):
        mean=(mean+df_0[index_0]['delta_line'].mean())
    mean=mean/len(df_0)
    for index_0 in range(len(df_0)):
        df_0[index_0].loc[(df_0[index_0]['delta_line']<0),'delta_line']=mean
    container_delta_line=np.array([],dtype=np.uint8)
    for df_temp in df_0:
        container_delta_line=np.append(container_delta_line,df_temp['delta_line'].values)
    if container_delta_line.max()!=container_delta_line.min() and container_delta_line.max()-container_delta_line.min()<6:
        for df_temp in df_0:
            df_temp['delta_line']=[stats.mode(container_delta_line)[0][0]]*df_temp.shape[0]
        
        
    for df_temp in df_0:
        df_temp['font_scaling']=df_temp['delta_line'].divide(container_delta_line.mean())
    
    
    return df_0

def id_clefs(df_0):
    
    df_1=df_0[0].copy()
    for index_0 in range(len(df_0)-1):
        df_1=df_1.append(df_0[index_0+1].copy())

    container_pass_counts=np.unique(df_1['pass_count'].values)
    container_clef_options=['g_clef','c_clef','f_clef']
    current_clefs=[]
    for clef in container_clef_options:
        if df_1.loc[df_1['kind']==clef].shape[0]!=0:
            current_clefs.append(clef)
    
    df_2=df_1.loc[df_1['kind']=='g_clef']
    df_3=pd.DataFrame(data={'kind':['g_clef'],'height':np.array([df_2['height'].mean()],dtype=np.uint16),'width':np.array([df_2['width'].mean()],dtype=np.uint16),'pixel_mean':np.array([df_2['pixel_mean'].mean()],dtype=np.uint16),'font_scaling':np.array([df_2['font_scaling'].values[0]])})
    
    if len(current_clefs) > 1:
        current_clefs=np.delete(current_clefs,'g_clef')
        for clef in current_clefs:
            df_2=df_1.loc[df_1['kind']==clef].copy()
            df_3=df_3.append(pd.DataFrame(data={'kind':[clef],'height':np.array([df_2['height'].mean()],dtype=np.uint16),'width':np.array([df_2['width'].mean()],dtype=np.uint16),'pixel_mean':np.array([df_2['pixel_mean'].mean()],dtype=np.uint16),'font_scaling':np.array([df_2['font_scaling'].values[0]])}))
    
    if len(container_pass_counts) > 1:
        for pass_count in container_pass_counts:
            if pass_count >= 0 and pass_count<99:
                df_2=df_1.loc[df_1['pass_count']==pass_count].copy()
                df_3=df_3.append(pd.DataFrame(data={'kind':[pass_count],'height':np.array([df_2['height'].mean()],dtype=np.uint16),'width':np.array([df_2['width'].mean()],dtype=np.uint16),'pixel_mean':np.array([df_2['pixel_mean'].mean()],dtype=np.uint16),'font_scaling':np.array([df_2['font_scaling'].values[0]])}))
    
    container_missing=np.unique(df_1.loc[df_1['pass_count']==99]['kind'])
    for index_0 in range(len(container_missing)):
        df_2=df_1.loc[df_1['kind']==container_missing[index_0]]
        df_3=df_3.append(pd.DataFrame(data={'kind':['missing_'+str(index_0)],'height':np.array([df_2['height'].mean()],dtype=np.uint16),'width':np.array([df_2['width'].mean()],dtype=np.uint16),'pixel_mean':np.array([df_2['pixel_mean'].mean()],dtype=np.uint16),'font_scaling':np.array([df_2['font_scaling'].values[0]])}))
    
    df_3=df_3.sort_values(by=['height'],ascending=[False])
    df_4=df_3.copy()
    df_4[['height','width']]=df_4[['height','width']].divide(df_4['font_scaling'],axis='index')
    df_4=df_4.reset_index(drop=True)
#    print(df_4)
    df_4.loc[(df_4['height']>df_4['height'].max()*0.75),'kind']='g_clef'
    df_4.loc[(df_4['height']<df_4['height'].max()*0.75),'kind']='f_clef'
    if df_4.loc[df_4['kind']=='f_clef'].shape[0]>1:
        if df_4.loc[df_4['kind']=='f_clef']['pixel_mean'].max()/df_4.loc[df_4['kind']=='f_clef']['pixel_mean'].min()>1.4:
#            print(df_4['pixel_mean'].astype(float),[df_4.loc[df_4['kind']=='f_clef']['pixel_mean'].max()*0.8]*df_4.shape[0])
#            print(df_4['kind'].astype(str),['g_clef']*df_4.shape[0])
            df_4['kind']=np.where((df_4['pixel_mean'].astype(float)>[df_4.loc[df_4['kind']=='f_clef']['pixel_mean'].max()*0.8]*df_4.shape[0]) & (df_4['kind'].astype(str)!=['g_clef']*df_4.shape[0]),'c_clef',df_4['kind'])
            #            df_4.loc[(df_4['pixel_mean']>df_4.loc[df_4['kind']=='f_clef']['pixel_mean'].max()*0.8 and df_4['kind']!='g_clef'),'kind']='c_clef'
    else:
        print('gottaCheck')
        
    df_3['id']=df_4['kind'].values
    df_3=df_3.reset_index(drop=True)
#    print(df_3)
    container_pass_index=[]
    for index,row in df_3.iterrows():
        if type(row['kind']) != str:
            container_pass_index.append(index)
    df_5=df_3.iloc[container_pass_index, :].copy()
    for index,row in df_5.iterrows():
        df_1.loc[(df_1['pass_count']==row['kind']),'kind']=row['id']
    container_missing=[]
    container_missing_id=[]
    for index,row in df_3.iterrows():
        if type(row['kind'])==str:
            if row['kind'][0]=='m':
                container_missing.append(row['kind'])
                container_missing_id.append(row['id'])
    container_fill=[]
    for index,row in df_1.iterrows():
        if type(row['kind'])==str:
            if row['kind'][0]=='f' and row['kind'][1]!='_':
                container_fill.append(row['kind'])
#    print(container_fill,container_missing)
    if len(container_fill) > 0 and len(container_missing) > 0:
#        print(np.unique(container_fill),np.unique(container_missing))
        df_6=pd.DataFrame(data={'missing':np.unique(container_missing),'fill':np.unique(container_fill)})
        container_missing_index,container_fill_index=[],[]
        for index,row in df_6.iterrows():
            container_fill_index.append([int(row['fill'][13]),int(row['fill'][15])])
            container_missing_index.append(int(row['missing'][8]))
        df_6['missing_index']=container_missing_index
        df_6['fill_index']=container_fill_index
        df_6['id']=container_missing_id
        df_7=df_1.sort_values(by=['y'],ascending=[True]).copy()
        df_7=df_7.reset_index(drop=True)
#        print(df_6)
        for index,row in df_6.iterrows():
            df_7.loc[(df_7['kind']==row['fill']),'kind']=row['id']
    else:
        df_7=df_1.copy()
    return df_7

def identify_noteheads(df_0,df_1,img):
    for index_0 in range(df_1.shape[0]):
        img_dup=img.copy()
        img_dup_1=img.copy()
        df_2=df_0.loc[df_0['yC']>df_1['y'].values[index_0]].copy()
        df_2=df_2.loc[df_2['yC']<df_1['y'].values[index_0]+df_1['height'].max()]
        df_2=df_2.loc[df_2['xC']>df_1['x'].values[index_0]]
        df_2=df_2.loc[df_2['height']>3.5*df_1['delta_line'].values[index_0]]
        print(df_2.shape[0])
        df_2=df_2.loc[df_2['ratio']>1.5]
        df_2=df_2.loc[df_2['ratio']<5]
        print(df_2.shape[0])
        df_2=df_2.sort_values(by=['x0'],ascending=[False])
        for index,row in df_2.iterrows():
            next_stage=False
            img_dup_blr=cv2.GaussianBlur(img,(5,5),0)
            template=img[int(row['y0']-5):int(row['y1']+5),int(row['x0']-5):int(row['x1']+5)]
            
#            cv2.imshow('template',template)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
            template_blr=cv2.GaussianBlur(template,(5,5),0)         
            res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    
            resolution=0.035 
            match_locations=np.where(res<=resolution)
            df_3=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
            df_3=df_3.sort_values(by=['x'],ascending=[True])
            df_3['delta_x']=df_3['x'].diff().shift().fillna(df_3['x'].diff().shift(-1))
            df_3=df_3.loc[df_3['delta_x']>2]
            df_3=df_3.sort_values(by=['y'],ascending=[True])
            df_3['delta_y']=df_3['y'].diff().shift().fillna(df_3['y'].diff().shift(-1))
            df_3=df_3.loc[df_3['delta_y']>2]
            
            if df_3.shape[0] > 3:
                split=int(template.shape[0]/2)
                next_stage=True
                break
                
        if df_2.shape[0]>1 and next_stage==True:
            if np.sum(template[:split,:]) > np.sum(template[split:,:]):
                template_split=template[split:,:].copy()
            else:
                template_split=template[:split,:].copy()
            
#            cv2.imshow('template_split',template_split)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
            bounds_vertical=np.where(template_split[:,int(template.shape[1]/2):int((template_split.shape[1]/2)+1)]<20)
            df_4=pd.DataFrame(data={'y':bounds_vertical[0]})
            df_4['delta_y']=df_4['y'].diff().shift().fillna(df_4['y'].diff().shift(-1))
            df_4['numrow']=df_4.index.tolist()
            df_4=df_4.loc[df_4['delta_y']>0]
            df_4['delta_numrow']=df_4['numrow'].diff().shift().fillna(df_4['numrow'].diff().shift(-1))
            df_4=df_4.loc[df_4['delta_numrow']<2]
            template_split_vert=template_split[df_4['y'].min():df_3['y'].max(),:].copy()
            df_5=pd.DataFrame(data={'col_0':template_split_vert[:,1].copy(),'col_1':template_split_vert[:,-1].copy()})
            df_5=df_5.divide(2)
            df_5['numrow']=df_5.index.tolist()
            df_5['sum']=df_5['col_0'].add(df_5['col_1'])
            df_6=df_5.loc[df_5['sum']<200].copy()
            df_6['delta_numrow']=df_6['numrow'].diff().shift().fillna(df_6['numrow'].diff().shift(-1))
            df_7=df_5.loc[df_5['sum']>=200].copy()
            df_7=df_7.reset_index(drop=True)
            df_7['delta_numrow']=df_7['numrow'].diff().shift(-1).fillna(df_7['numrow'].diff().shift())
            if df_6.loc[df_6['delta_numrow']>1].shape[0] > 1:
                print('LINE')
#                print(df_7)
            else:
                #if on a space
#                print('SPACE',df_7['delta_numrow'].idxmax(),df_7['numrow'].values[df_7['delta_numrow'].idxmax()+1])
#                print(df_7)
                if np.abs(df_7['numrow'].values[0]-df_7['numrow'].values[df_7['delta_numrow'].idxmax()]) > np.abs(df_7['numrow'].values[-1]-df_7['numrow'].values[df_7['delta_numrow'].idxmax()+1]):
                    template_trimed_vert=template_split_vert[df_7['numrow'].values[0]:df_7['numrow'].values[df_7['delta_numrow'].idxmax()],:].copy()
                else:
                    template_trimed_vert=template_split_vert[df_7['numrow'].values[df_7['delta_numrow'].idxmax()+1]:df_7['numrow'].values[-1],:].copy()
                
                
                
#                cv2.imshow('template_trimed_vert',template_trimed_vert)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
                
                note_location=np.where(template_trimed_vert<=20)
                df_8=pd.DataFrame(data={'x':note_location[1],'y':note_location[0]})
                
                template_trimmed_hor=template_trimed_vert[df_8['y'].min():df_8['y'].max(),df_8['x'].min():df_8['x'].max()]
#                cv2.imshow('template_trimmed_hor',template_trimmed_hor)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
            
            
#            print(df_5)
#            print(df_3)
#            template_split_vert=template_split[:,np.min(bounds_vertical):np.max(bounds_vertical)].copy()
            
#            cv2.imshow('template',template_split_vert)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
                img_dup_blr=cv2.GaussianBlur(img,(13,13),0)
                th,img_dup_th=cv2.threshold(img_dup_blr,225,255,cv2.THRESH_BINARY_INV)
                template_blr=cv2.GaussianBlur(template_trimmed_hor,(13,13),0)
                th,template_th=cv2.threshold(template_blr,225,255,cv2.THRESH_BINARY_INV)
#                
#                cv2.imshow('img_dup_th',img_dup_th)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
             
                res=cv2.matchTemplate(img_dup_th,template_th,eval('cv2.TM_SQDIFF_NORMED'))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
                resolution=0.045
     
                match_locations=np.where(res<=resolution)
                df_9=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
                for index,row in df_9.iterrows():
                    cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+template_trimmed_hor.shape[1]),(int(row['y']+template_trimmed_hor.shape[0]))),[155,155,155],2)
                cv2.imwrite('result_0.png',img_dup)
                print(df_9.shape[0],'init')
                break
    
    container_noteheads=[]
    for index_0 in range(df_1.shape[0]):
        if index_0!=0 and index_0!=df_1.shape[0]-1:
            print(container_noteheads[index_0-1]['y'].max(),df_1['y1'].values[index_0]+((df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2))
            container_noteheads.append(df_9.loc[df_9['y']>container_noteheads[index_0-1]['y'].max()].copy())
            container_noteheads[index_0]=container_noteheads[index_0].loc[container_noteheads[index_0]['y']<df_1['y1'].values[index_0]+((df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2)]
        elif index_0 == df_1.shape[0]-1:
            print('>',container_noteheads[index_0-1]['y'].max())
            container_noteheads.append(df_9.loc[df_9['y']>container_noteheads[index_0-1]['y'].max()].copy())
        elif index_0 == 0:
            print('<',df_1['y1'].values[index_0]+((df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2))
            container_noteheads.append(df_9.loc[df_9['y']<df_1['y1'].values[index_0]+((df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2)].copy())
    count=0        
    for index_1 in range(len(container_noteheads)):
        container_noteheads[index_1]=container_noteheads[index_1].sort_values(by=['x'],ascending=[True])
        container_noteheads[index_1]['delta_x']=container_noteheads[index_1]['x'].diff().shift(-1).fillna(10)
#        print(container_noteheads[index_0])
        container_noteheads[index_1]=container_noteheads[index_1].loc[container_noteheads[index_1]['delta_x']>9]
#        print(container_noteheads[index_1],'per')
        count=count+container_noteheads[index_1].shape[0]
        for index_1,row in container_noteheads[index_1].iterrows():
            cv2.rectangle(img_dup_1,(int(row['x']),int(row['y'])),(int(row['x']+template_trimmed_hor.shape[1]),(int(row['y']+template_trimmed_hor.shape[0]))),[155,155,155],15)
    cv2.imwrite('result_1.png',img_dup_1)
    print(count)
    return container_noteheads

def locate_staff_contents(df_0,df_1,img):
    df_1=df_1.reset_index(drop=True)
    img_dup_0=img.copy()
    img_dup_1=img.copy()
    img_dup_2=img.copy()
    img_dup_3=img.copy()
    for index_0,row in df_1.iterrows():
        if index_0==0:
            container_content_bounds=df_0.loc[df_0['y0']<df_1['y'].values[index_0+1]+(df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2].copy()
            container_content_bounds=container_content_bounds.loc[container_content_bounds['y0']>df_1['y'].min()-(2*df_1['delta_line'].mean())]
        elif index_0==df_1.shape[0]-1:
            container_content_bounds=container_content_bounds.append(df_0.loc[df_0['y0']>container_content_bounds['y0'].max()].copy())
            container_content_bounds=container_content_bounds.loc[container_content_bounds['y0']<df_1['y'].max()+(8*df_1['delta_line'].mean())]
        else:
            df_2=df_0.loc[df_0['y0']>container_content_bounds['y0'].max()].copy()
            df_2=df_2.loc[df_2['y0']<df_1['y'].values[index_0+1]+(df_1['y'].values[index_0+1]-df_1['y1'].values[index_0])/2]
            container_content_bounds=container_content_bounds.append(df_2.copy())
    container_content_bounds=container_content_bounds.loc[container_content_bounds['x0']>df_1['x'].values[index_0]+df_1['width'].values[index_0]]        
    container_content_bounds=container_content_bounds.sort_values(by=['area'],ascending=[False])
    container_content_bounds['delta_area']=container_content_bounds['area'].diff().shift(-1).abs().fillna(container_content_bounds['area'].diff().shift().abs())
    container_content_bounds=container_content_bounds.loc[container_content_bounds['delta_area']>5]
    container_content_bounds=container_content_bounds.sort_values(by=['area'],ascending=[True])
    
    df_3=container_content_bounds.loc[container_content_bounds['ratio']>=0.66].copy()
    df_3=df_3.loc[df_3['area']>df_1['delta_line'].mean()**2.33]
    df_3=df_3.loc[df_3['width']>df_1['delta_line'].mean()*0.5]
    df_3=df_3.reset_index(drop=True)
    df_4=container_content_bounds.loc[container_content_bounds['ratio']<=0.66].copy()
    
    for index_0,row in df_3.iterrows():
        if index_0 < 40:
            
            img_dup_0=img.copy()
            img_dup_1=img.copy()
            instance=img[int(row['y0']):int(row['y1']),int(row['x0']):int(row['x1'])].copy()
            
            th,instance_th=cv2.threshold(instance,135,255,cv2.THRESH_BINARY)
            w=int(instance_th.shape[1]*3/5)
            if w%2!=1:
                w=w-1
            instance_th_blr=cv2.GaussianBlur(instance_th,(w,3),0)
            th,instance_th_blr_th=cv2.threshold(instance_th_blr,135,255,cv2.THRESH_BINARY)
            locations=np.where(instance_th_blr_th<45)
            df_5=pd.DataFrame(data={'x':locations[1],'y':locations[0]})
            instance_trim=instance[df_5['y'].min():df_5['y'].max(),df_5['x'].min():df_5['x'].max()]
            instance_th_trim=instance_th_blr_th[df_5['y'].min():df_5['y'].max(),df_5['x'].min():df_5['x'].max()]
            
            instance_th_trim_padded=cv2.copyMakeBorder(instance_th_trim,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255,255])
            hold_0,contours_0,hierarchy=cv2.findContours(instance_th_trim_padded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_0:
                x,y,w,h=cv2.boundingRect(c)
                if h > instance_trim.shape[0]*3/5 and h < instance_trim.shape[0]*11/10:
                    instance_bound = instance_trim[y-49:y+h-49,x-49:x+w-49]
                    instance_th_bound = instance_th_trim[y-49:y+h-49,x-49:x+w-49]
            locations=[]
            for index_1 in range(instance_th_bound.shape[0]):
                if np.mean(instance_th_bound[index_1,:])<45:
                    locations.append(index_1)
            df_6=pd.DataFrame(data={'val':locations})
            df_7=df_6.copy()
            df_7['delta_val']=df_7['val'].diff().shift(-1).abs().fillna(df_7['val'].diff().shift().abs())
            df_7=df_7.loc[df_7['delta_val']>1]
            df_8=df_6.copy()
            df_8['delta_val']=df_8['val'].diff().shift().abs().fillna(df_8['val'].diff().shift(-1).abs())
            df_8=df_8.loc[df_8['delta_val']>1]
            
            top,bottom=0,instance_th_bound.shape[0]
#            print('org: ',top,bottom)
            
            
            if df_6['val'].min() < 8:
                if df_7.shape[0] > 1:
                    top=int(df_7['val'].min()+2)
                    print(top,df_7)
                else:
                    top=int(df_6.min()+2)
#                    print(top,'df_6')
            if df_6['val'].max() > instance_th_bound.shape[0]-8:
                if df_8.shape[0] > 1:
                    bottom=int(df_8['val'].max()-2)
#                    print(bottom,df_8)
                else:
                    bottom=int(df_6['val'].max()-2)
#                    print(bottom,'df_6')
#            print('final: ',top,bottom)
            if top >=0 and bottom >=0:
                instance_bound_cleaned=instance_bound[top:bottom,:].copy()
                instance_th_bound_cleaned=instance_th_bound[top:bottom,:].copy()

                res=cv2.matchTemplate(img_dup_0,instance_bound_cleaned,eval('cv2.TM_SQDIFF_NORMED'))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
                match_locations=np.where(res<=0.035)
                df_9=pd.DataFrame(data={'x':np.array(match_locations[1],dtype=np.uint16),'y':np.array(match_locations[0],dtype=np.uint16)})
                
                for index_1,row in df_9.iterrows():
                    cv2.rectange(img_dup_1,(row['x'],row['y']),(instance_bound_cleaned.shape[1],instance_bound_cleaned.shape[0]),[155,155,155],2)
                
                cv2.imwrite('result_'+str(index_0)+'.png',img_dup_1)
#                cv2.imshow('instance_bound_cleaned',instance_bound_cleaned)
#                cv2.imshow('instance_th_bound_cleaned',instance_th_bound_cleaned)
##                cv2.imshow('instance',instance)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
        
    return container_content_bounds





#fldr_name='Bach, Johann Sebastian_Sonata I BWV 1001'
path= 'source/scores/'
#a,b=fldr_name.split('_')
#score={'page_count':len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]),'composer':a,'title':b,'voices':[]}
df_0,img=init_img_filter(path+'img_49.png')
df_1=find_g_clefs(df_0,img)
df_2=find_clefs_from_reference(df_0,df_1,img)
df_2=calc_staff_font_info(df_2,img)
#print('system_count: ',len(df_2))
df_3=id_clefs(df_2)
df_3=df_3.sort_values(by=['y'],ascending=[True])
#print(df_3)
#df_4=identify_noteheads(df_0,df_3,img)
df_4=locate_staff_contents(df_0,df_3,img)


#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def init_score(fldr_name):
    path= 'source/scores/'+fldr_name+'/'
    a,b=fldr_name.split('_')
    score={'page_count':len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]),'composer':a,'title':b,'voices':[]}
    df_0,img=init_img_filter(path+'img_0.png')
    df_1=find_g_clefs(df_0,img)
    df_2=find_clefs_from_reference(df_0,df_1,img)
    df_2=calc_staff_font_info(df_2,img)
    print('system_count: ',len(df_2))
    df_3=id_clefs(df_2)
    df_3=df_3.sort_values(by=['y'],ascending=[True])
    print(df_3)
    df_4=identify_noteheads(df_0,df_3,img)
    
    '''
    OCR THINGS BELOW 
    parse_title_text=img[:int(df_3['y'].values[0]+df_3['height'].values[0]*0.33),:]
    cv2.imshow('title?',parse_title_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test_img=Image.fromarray(parse_title_text)
    text=pytesseract.image_to_string(test_img,lang='rus')    
    print(text)
    df_parsing=pd.DataFrame(data={'content':text.split('\n')})
    df_parsing=df_parsing.loc[df_parsing['content'].str.len()>1]
    
    text=pytesseract.image_to_string(test_img,lang='eng')
    df_parsing=df_parsing.append(pd.DataFrame(data={'content':text.split('\n')}))
    df_parsing=df_parsing.loc[df_parsing['content'].str.len()>1]

#    print(text)
    '''
    return df_parsing
    
#title_text=init_score('Rachmaninoff, Sergei_Bogorditse Devo Opus 37')

def find_noteheads_in_systems(df_0,df_1):
    
    for index_0 in range(len(df_1)):
        for index_1 in range(df_1[index_0].shape[0]):
#            print(df_1[index_0].iloc[index_1:index_1+1]['y'].values-25)
            df_2=df_0.loc[df_0['yC']>df_1[index_0].iloc[index_1:index_1+1,:]['y'].values[0]].copy()
            df_2=df_2.loc[df_2['yC']<df_1[index_0].iloc[index_1:index_1+2,:]['y1'].values[0]]
            df_2=df_2.loc[df_2['x0']>df_1[index_0].iloc[index_1:index_1+1,:]['x'].values[0]+75]
            df_2=df_2.sort_values(by=['x0'],ascending=[True])
            df_2=df_2.loc[df_2['height']>df_1[index_0]['height'].min()*0.85]
            df_2=df_2.loc[df_2['height']<df_1[index_0]['height'].max()*1.33]
            df_2=df_2.loc[df_2['ratio']>1.25]
            for index_2 in range(df_2.shape[0]):
                
                info=df_2.iloc[index_2:index_2+1,:]
                print(info)
                template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
#                template_blr=cv2.GaussianBlur(template,(w,h),0)
                
                cv2.imshow('template',template)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
          
    return df_1

#def multithread_test(img):
#    print(img)
#    df_0,img=init_img_filter(img)
#    df_1=find_g_clefs(df_0,img)
#    df_2=find_clefs_from_reference(df_0,df_1,img)
##    df_3=calc_staff_font_info(df_2)
#    
#    return df_0,df_1,df_2,df_3
'''
results=[]
for val in [19,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]:
    
    print('in : ',val)
    df_0,img=init_img_filter('source/scores/img_'+str(val)+'.png')
#    print('done0')
    df_1=find_g_clefs(df_0,img)
#    print('done1')
    df_2=find_clefs_from_reference(df_0,df_1,img)
#    print('done2')
    df_2=calc_staff_font_info(df_2)
#    for index_0 in range(len(df_2)):
#        print(df_2[index_0]['font_scaling'].max(),df_2[index_0]['font_scaling'].min())
    df_3=id_clefs(df_2)
    results.append(df_3)
#    print(df_3['kind'].tolist())
#    valu=input('y/n')
#    if valu=='n':
#        print(val)
#        break
'''
'''
21 c-clef miss ID at pass_count stage
36 f-clef miss ID at find_g_clefs stage
45 duplicate line at find_from_reference
47 Issue with missing clefs, but I have the template so I believe it has to do with filtering
62 Could not locate non-g_clefs
63 find g_clefs did not locate all and that let to issues, I need to look at the find other clefs 
    parameters because there are a very large difference between delta_y's
65 not all clefs IDed
'''
#from multiprocessing.dummy import Pool as ThreadPool
#pool=ThreadPool(4)
#results = pool.map(multithread_test,['source/scores/img_21.png','source/scores/img_22.png','source/scores/img_21.png','source/scores/img_22.png'])
    
    
    
    
'''
Bach, Johann Sebastian_Sonata I BWV 1001
Rachmaninoff, Sergei_Bogoroditse Devo Opus 37
Strauss, Richard_Divertimento op 86
'''
    
#csv_test=pd.read_csv('source/Music Dictionary - Sheet1.csv')


