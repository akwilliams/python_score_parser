import cv2
import numpy as np
import pandas as pd
from scipy import stats
import os

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
#    container_mins=np.array([],dtype=np.float32)
#    for index_0 in range(df_0['x0'].mode().shape[0]):
#        container_mins=np.append(container_mins,[df_0.loc[df_0['x0']==df_0['x0'].mode()[index_0]]['x0'].min()])
#    df_0=df_0.loc[df_0['x0']==df_0['x0'].mode()[np.argmin(container_mins)]]
    df_0=df_0.sort_values(by=['x0'],ascending=[True])
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
    df_1=df_1.reset_index(drop=True)
    
#    if df_1.shape[0]>0:
#        w,h=df_0['width'].values[0],df_0['height'].values[0]
#        img_dup=img.copy()
#        for index,row in df_1.iterrows():
#            cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+w),int(row['y']+h)),[155,155,155],2)
#        
#        cv2.imshow('thing',img_dup)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
      
    if df_1['x'].max()-df_1['x'].min()>1000:
        df_1=df_1.loc[df_1['x']<df_1['x'].mean()]
    
    
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
        if area>=(height*width*0.025) and area<=(height*width):
            x,y,w,h=cv2.boundingRect(c)
            container_bounds=np.append(container_bounds,[y,y+h])
    
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
#        print(df_5)
        container_df.append(df_5.copy())
        
      

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
#        print(df_7)
        if df_7['delta_y'].min()>108:
            df_6=df_6.append(container_df[index_0].copy())
            df_6['pass_count']=df_6['pass_count'].fillna(index_0)
            df_6=df_6.sort_values(by=['y'],ascending=[True])
#            print(df_6)
    df_6=df_6.sort_values(by=['y'],ascending=[True])
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(df_6['y'].diff().shift())
    df_6=df_6.loc[df_6['y']<container_bounds.max()+100]
    df_6=df_6.loc[df_6['y']>container_bounds.min()-100]
    
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(0)
    minimum,mean,maximum=df_6.iloc[:-1,:]['delta_y'].min(),df_6.iloc[:-1,:]['delta_y'].mean(),df_6.iloc[:-1,:]['delta_y'].max()
    df_6=df_6.reset_index(drop=True)
#    print(df_6)
    if (maximum-mean)/mean>0.5:
        
        for index_0 in range(5):
            if (maximum-mean)/mean>0.5:
                location=df_6['delta_y'].idxmax(axis=1)
                df_8=df_6.iloc[location:location+2].copy()
                df_9=df_0.loc[df_0['y0']>df_8['y'].min()+(minimum*0.1)].copy()
                df_9=df_9.loc[df_9['y1']<df_8['y'].max()-(minimum*0.1)]
                df_9=df_9.loc[df_9['x0']>df_8['x'].min()-20]
                df_9=df_9.loc[df_9['x0']<df_8['x'].max()+20]
                df_9=df_9.loc[df_9['area']>100]
                df_9=df_9.sort_values(by=['area'],ascending=[False])
                df_9=df_9.reset_index(drop=True)
#                print(df_9)
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
                    minimum,mean,maximum=df_6.iloc[:-1,:]['delta_y'].min(),df_6.iloc[:-1,:]['delta_y'].mean(),df_6.iloc[:-1,:]['delta_y'].max()
                    df_6=df_6.reset_index(drop=True)
                    break
            else:
#                print('off')
                break
    df_6['delta_y']=df_6['y'].diff().shift(-1).fillna(1)              
    df_6['y1']=df_6['y'].add(df_6['height'])
    df_6['delta_y1']=df_6['y'].subtract(df_6['y1'].shift()).fillna(1)
    df_6=df_6.loc[df_6['delta_y1']>-2]
    df_6=df_6.loc[df_6['delta_y1']<750]
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
        container_font_info={'pixel_mean':[],'delta_line':[],'pass_count':[],'kind':[]}
        for index_1 in range(df_0[index_0].shape[0]):
            info=df_0[index_0].iloc[index_1:,:].copy()
            if info['height'].values[0]<100:
                height=df_0[index_0]['height'].max()*0.66
                w,h=int(info['width'].values[0]),1
                template=img[info['y'].values[0]:int(info['y'].values[0]+height),info['x'].values[0]-1:info['x'].values[0]+info['width'].values[0]+4]            
            else:
                w,h=int(info['width'].values[0]),1
                template=img[info['y'].values[0]:info['y1'].values[0],info['x'].values[0]-1:info['x'].values[0]+info['width'].values[0]+4]
            if w%2!=1:
                w=w-1
                if w<=0:
                    w=1
            if h%2!=1:
                h=h-1
                if h<=0:
                    h=1
#            if w > 255:
#                w=201
#            print(w,h,template.shape)
            template_blr=cv2.GaussianBlur(template,(w,h),0)
#            cv2.imshow('template',template)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            th,template_th=cv2.threshold(template_blr,165,255,cv2.THRESH_BINARY_INV)
            df_1=pd.DataFrame(data={'row_0':template_th[:,0].copy(),'row_1':template_th[:,-1].copy()})
            
                
            df_1=df_1.divide(2)
            df_1['sum']=df_1['row_0'].add(df_1['row_1'])
#            print(df_1)
            df_1=df_1.loc[df_1['sum']>200]
#            print(df_1)
            df_1['numrow']=df_1.index.tolist()
            df_1['delta_p']=df_1['numrow'].diff().shift(-1).fillna(2)
            df_1=df_1.loc[df_1['delta_p']>5]
#            print(df_1)
            df_1['delta_line']=df_1['numrow'].diff().shift(-1).fillna(df_1['numrow'].diff().shift().fillna(df_1['delta_p'].max()))
            container_font_info['pixel_mean'].append(np.mean(template_th))
#            print(df_1)
            if np.abs(df_1['delta_line'].max()-df_1['delta_line'].min()) > 6:
                container_font_info['delta_line'].append(df_1['delta_line'].tolist())
                needs_postprocessing=True
            else:
                container_font_info['delta_line'].append(int(df_1['delta_line'].mean()))  
            
            
        df_0[index_0]['delta_line']=container_font_info['delta_line']
        df_0[index_0]['pixel_mean']=container_font_info['pixel_mean']
#    for index_0 in range(len(df_0)):
#        print(df_0[index_0])
#    print(df_0)
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
        df_0[data[2]]['delta_line'].values[data[1]]=np.mean(df_0[data[2]]['delta_line'].values[data[1]])
    container_df_length=[]
    for index_0 in range(len(df_0)):
        container_df_length.append(df_0[index_0].shape[0])
    if len(np.unique(container_df_length))==1:
        container_d_line=[]
        container_corrected_d_line=[]
        for index_0 in range(len(df_0)):
            for index_1 in range(len(df_0[index_0]['delta_line'].tolist())):
                if index_0==0:
                    container_d_line.append([df_0[index_0]['delta_line'].tolist()[index_1]])
                else:
                    container_d_line[index_1].append(df_0[index_0]['delta_line'].tolist()[index_1])
#            print(container_d_line)
        for index_0 in range(len(container_d_line)):
            container_corrected_d_line.append(np.mean(container_d_line[index_0]))
        for index_0 in range(len(df_0)):
            for index_1 in range(df_0[index_0].shape[0]):
                df_0[index_0]['delta_line']=container_corrected_d_line
#                print(df_0[index_0])
    
    
    
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
    if df_4['pixel_mean'].max()>120:
        df_4.loc[(df_4['pixel_mean']>df_4['pixel_mean'].mean()*1.2),'kind']='c_clef'
    df_3['id']=df_4['kind'].values
    df_3=df_3.reset_index(drop=True)
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
        for index,row in df_6.iterrows():
            df_7.loc[(df_7['kind']==row['fill']),'kind']=row['id']
    else:
        df_7=df_1.copy()
    return df_7


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
#                print(info)
                template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
#                template_blr=cv2.GaussianBlur(template,(w,h),0)
                
                cv2.imshow('template',template)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
          
    return df_1

def init_score(fldr_name):
    path = 'source/scores/'+fldr_name+'/'
    a,b=fldr_name.split('_')
    score={'page_count':len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]),'composer':a,'title':b,'voices':[]}
    df_0,img=init_img_filter(path+'img_0.png')
    df_1=find_g_clefs(df_0,img)
    df_2=find_clefs_from_reference(df_0,df_1,img)
    df_2=calc_staff_font_info(df_2,img)
    df_3=id_clefs(df_2)
    print(df_3)
    return fldr_name
#def multithread_test(img):
#    print(img)
#    df_0,img=init_img_filter(img)
#    df_1=find_g_clefs(df_0,img)
#    df_2=find_clefs_from_reference(df_0,df_1,img)
##    df_3=calc_staff_font_info(df_2)
#    
#    return df_0,df_1,df_2,df_3
'''
26 fill missing issue
'''

for val in [19,20,21,22,23,25,27,28,29,30,31,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65]:

    df_0,img=init_img_filter('source/scores/img_'+str(val)+'.png')
    print('init_img_filter')
    df_1=find_g_clefs(df_0,img)
    print('find_g_clefs')
    df_2=find_clefs_from_reference(df_0,df_1,img)
    print('find_clefs_from_ref')
    df_2=calc_staff_font_info(df_2,img)
#    print(df_2[0]['delta_line'],df_2[0]['font_scaling'])
    df_3=id_clefs(df_2)
    print(df_3)
    
#name=init_score('Brahms, Johannes_11 Chorale Prelueds')
    
    
#from multiprocessing.dummy import Pool as ThreadPool
#pool=ThreadPool(4)
#results = pool.map(multithread_test,['source/scores/img_21.png','source/scores/img_22.png','source/scores/img_21.png','source/scores/img_22.png'])
    
    
    
    
    
    
    


