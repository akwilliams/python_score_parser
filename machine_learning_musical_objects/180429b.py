import cv2
import numpy as np
import pandas as pd

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
    return df,img


def find_g_clefs(df,img):
    df_0=df.loc[df['ratio']>2.5]
    df_0=df_0.loc[df_0['ratio']<3.1]
    df_0=df_0.loc[df_0['area']<25000]
    df_0=df_0.loc[df_0['area']>3000]
    
    df_0=df_0.sort_values(by=['area'],ascending=[False])
    
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q1']]
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q2']]
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q3']]
    
    df_0=df_0.loc[df_0['x0']<df_0['x0'].min()*2]
    if df_0['x0'].max()!=df_0['x0'].min():
        df_0=df_0.loc[df_0['x0']<df_0['x0'].max()]
        
        
    count = df_0.shape[0]
    if count>6:
        count = 6
    
    for index_0 in range(count):
        info = df_0.iloc[index_0:,:]
        w,h=int(info['width'].tolist()[0]*0.25),int(info['height'].tolist()[0]*0.25)
        if w%2!=1:
            w=w-1
        if h%2!=1:
            h=h-1
            
        img_dup=img.copy()
        img_dup_blr=cv2.GaussianBlur(img_dup,(w,h),0)
        template=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
        template_blr=cv2.GaussianBlur(template,(w,h),0)
        
#        cv2.imshow('template',template)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    
        res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        resolution=0.035
        difference_boo=False
        
        for index_1 in range(7):
            match_locations=np.where(res<=resolution)
            resolution=resolution-0.005
            df_1=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
            length=df_1.shape[0]
            if length>0 and length<5000:
                df_2=df_1.copy()
                df_3=df_1.copy()
                df_2['delta_x']=df_2['x'].diff().shift(1).fillna(df_2['x'].diff().shift(-1))
                df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
                df_2=df_2.loc[df_2['delta_y']>1]
                df_2=df_2.append(df_1.iloc[0:1,:])
                df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
                df_2=df_2.loc[df_2['delta_y']>100]
                
                df_2=df_2.append(pd.DataFrame(data={'x':[0,0],'y':[0,img.shape[:2][0]]}))
                df_2=df_2.sort_values(by=['y'],ascending=[True])
                index_container=[]
                difference_boo=True
            
                for index_2 in range(df_2.shape[0]-1):
                    temp_0=df_3.loc[df_3['y']>=df_2.iloc[index_2:,:]['y'].tolist()[0]].copy().index.tolist()
                    temp_1=df_3.loc[df_3['y']<df_2.iloc[(index_2+1):,:]['y'].tolist()[0]].copy().index.tolist()
                    index_container.append([vari for vari in temp_0 if vari in temp_1])
            
                for instance in index_container:
                    if len(instance)>0 and difference_boo==True:
                        df_4=df_1.iloc[instance[0]:instance[-1],:].copy()
                        df_4['delta_x']=df_4['x'].diff().shift(1).fillna(df_4['x'].diff().shift(-1))
                        df_4['delta_y']=df_4['y'].diff().shift(1).fillna(df_4['y'].diff().shift(-1))        
                        df_4=df_4.loc[df_4['delta_y']>0]
                        if df_4['x'].max()-df_4['x'].min()>35:
                            difference_boo=False
            else:
                break
            if df_1.shape[0]<1 or difference_boo==True:
                break
        df_2=df_1.copy()
        df_2['delta_x']=df_2['x'].diff().shift(1).fillna(df_2['x'].diff().shift(-1))
        df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
        df_2=df_2.loc[df_2['delta_y']>1]
        df_2=df_2.append(df_1.iloc[0:1,:])
        df_2=df_2.sort_values(by=['y'],ascending=[True])
        if df_2.shape[0]>1 and difference_boo==True:
            break
    if df_2.shape[0]>0:
        w,h=info['width'].tolist()[0],info['height'].tolist()[0]
        img_dup=img.copy()
        for index,row in df_2.iterrows():
            cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+w),int(row['y']+h)),[155,155,155],2)
        
#        cv2.imshow('thing',img_dup)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
      
    length=df_2.shape[0]
    
    df_2['kind']=['g_clef']*length
    df_2['width']=[w]*length
    df_2['height']=[h]*length
    df_2['area']=df_2['width'].multiply(df_2['height'])
    
    return df_2

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
    
    #Here is where I need to parse df_6 to id C and F clefs
    container_system_info=[]
    system_count=container_bounds.shape[0]/2
    for index_0 in range(int(system_count)):
        y_bounds=container_bounds[index_0*2:(index_0*2)+2]
        df_15=df_6.loc[df_6['y']>y_bounds.min()-25].copy()
        df_15=df_15.loc[df_15['y']<y_bounds.max()+25]
        container_system_info.append(df_15.copy())
        
    return container_system_info

def calc_staff_font_info(df_0):
    for index_0 in range(len(df_0)):
        container_font_info={'pixel_mean':[],'delta_line':[]}        
        for index_1 in range(df_0[index_0].shape[0]):
            info=df_0[index_0].iloc[index_1:,:].copy()
            if info['height'].values[0]<100:
                height=df_0[index_0]['height'].max()*0.66
                w,h=int(info['width'].values[0]*0.05125),int(height*0.05125)
                if w%2!=1:
                    w=w-1
                    if w<=0:
                        w=1
                if h%2!=1:
                    h=h-1
                    if h<=0:
                        h=1
                
                template=img[info['y'].values[0]:int(info['y'].values[0]+height),info['x'].values[0]:info['x'].values[0]+info['width'].values[0]]
            
            else:
                w,h=int(info['width'].values[0]*0.05125),int(info['height'].values[0]*0.05125)
                if w%2!=1:
                    w=w-1
                    if w<=0:
                        w=1
                if h%2!=1:
                    h=h-1
                    if h<=0:
                        h=1
                template=img[info['y'].values[0]:info['y1'].values[0],info['x'].values[0]:info['x'].values[0]+info['width'].values[0]]
            template_blr=cv2.GaussianBlur(template,(w,h),0)
            th,template_th=cv2.threshold(template_blr,165,255,cv2.THRESH_BINARY_INV)
            cv2.imshow('template',template_th)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#            print(template_th.shape)
#            print(np.where(template_th[:,0]==255),np.where(template_th[:,-1]==255))
            df_1=pd.DataFrame(data={'row_0':template_th[:,0].copy(),'row_1':template_th[:,-1].copy()})
            df_1=df_1.divide(2)
            df_1['sum']=df_1['row_0'].add(df_1['row_1'])
            df_1=df_1.loc[df_1['sum']>200]
            df_1['numrow']=df_1.index.tolist()
            df_1['delta_p']=df_1['numrow'].diff().shift(-1).fillna(2)
            df_1=df_1.loc[df_1['delta_p']>5]
            df_1['delta_line']=df_1['numrow'].diff().shift(-1).fillna(df_1['numrow'].diff().shift())
            container_font_info['pixel_mean'].append(np.mean(template_th))
            container_font_info['delta_line'].append(int(df_1['delta_line'].mean()))
            
#            cv2.imshow('template',template_th)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        df_0[index_0]['delta_line']=container_font_info['delta_line']
        df_0[index_0]['pixel_mean']=container_font_info['pixel_mean']
        
        df_2=df_0[index_0].loc[df_0[index_0]['pass_count']>-1].copy()
        container_pass_count=df_2['pass_count'].value_counts().copy().index.tolist()
        container_averages=[]
        for index_2 in range(len(container_pass_count)):
            container_averages.append(df_2.loc[df_2['pass_count']==container_pass_count[index_2]]['pixel_mean'].mean())
        print(container_pass_count,container_averages)
        if len(container_pass_count)==1 and container_averages[0]<150:
            for index_2 in df_0[index_0].loc[df_0[index_0]['pass_count']==container_pass_count[0]].index.tolist()
#            for index_2 in df_0[index_0].loc[df_0[index_0]['pass_count']==container_pass_count[0]].index.tolist():
#                if(index_2+1<df_0[index_0].shape[0]):
#                    df_0[index_0].iloc[index_2:index_2+1,:]['kind'].values[0]='f_clef'
#                else:
#                    df_0[index_0].iloc[-1:,:]['kind'].values[0]='f_clef'
                
#        elif len(container_pass_count==2):
#            #code
            
#        print(df_2.index)
    '''To tell the difference between C clefs and F clef hold the pixel mean
    of each group and average, if there is a distict difference, the ones with
    the larger values are c clefs and the smaller F clefs'''
    
    return df_1
df_3=calc_staff_font_info(df_2)


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



for val in range(1):
#    import time
#    start=time.time()
    df_0,img=init_img_filter('source/scores/img_'+str(val+23)+'.png')
#    end=time.time()
#    print(end-start)
#    start=time.time()
    df_1=find_g_clefs(df_0,img)
#    end=time.time()
#    print(end-start)
#    start=time.time()
    df_2=find_clefs_from_reference(df_0,df_1,img)
#    end=time.time()
#    print(end-start)
    df_3=calc_staff_font_info(df_2)
#    df_4=find_noteheads_in_systems(df_0,df_2)
    
    
    
    
    
    
    
    
    
    


