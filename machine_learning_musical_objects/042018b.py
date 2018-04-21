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
    
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q1']]
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q2']]
    df_0=df_0.loc[df_0['pixel_mean_q0']>df_0['pixel_mean_q3']]
    
    df_0=df_0.loc[df_0['x0']<df_0['x0'].min()*2]
    if df_0['x0'].max()!=df_0['x0'].min():
        df_0=df_0.loc[df_0['x0']<df_0['x0'].max()]
        
    count = len(df_0.index.tolist())
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
        
        cv2.imshow('template',template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        res=cv2.matchTemplate(img_dup_blr,template_blr,eval('cv2.TM_SQDIFF_NORMED'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        resolution=0.035
        difference_boo=False
        
        for index_1 in range(7):
            match_locations=np.where(res<=resolution)
            resolution=resolution-0.005
            df_1=pd.DataFrame(data={'x':match_locations[1],'y':match_locations[0]})
            if len(df_1.index.tolist())>0 and len(df_1.index.tolist())<5000:
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
                print(df_2)
            
                for index_2 in range(len(df_2.index.tolist())-1):
                    temp_0=df_3.loc[df_3['y']>=df_2.iloc[index_2:,:]['y'].tolist()[0]].copy().index.tolist()
                    temp_1=df_3.loc[df_3['y']<df_2.iloc[(index_2+1):,:]['y'].tolist()[0]].copy().index.tolist()
                    index_container.append([vari for vari in temp_0 if vari in temp_1])
            
                for instance in index_container:
                    if len(instance)>0 and difference_boo==True:
                        df_4=df_1.iloc[instance[0]:instance[-1],:].copy()
                        df_4['delta_x']=df_4['x'].diff().shift(1).fillna(df_4['x'].diff().shift(-1))
                        df_4['delta_y']=df_4['y'].diff().shift(1).fillna(df_4['y'].diff().shift(-1))        
                        df_4=df_4.loc[df_4['delta_y']>0]
                        if df_4['x'].max()-df_4['x'].min()>35:# and df_4['y'].max()-df_4['y'].min()>35:
                            print('difference_boo is false')
                            difference_boo=False
            else:
                break
            if len(df_1.index.tolist())<1 or difference_boo==True:
                break
#                resolution=resolution-0.005
#                df_2=df_1.copy()
#                df_2['delta_x']=df_2['x'].diff().shift(1).fillna(df_2['x'].diff().shift(-1))
#                df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
#            df_2=df_2.loc[df_2['delta_y']>1]
#            df_2=df_2.append(df_1.iloc[0:1,:])
#            if len(df_2.index.tolist())>1:
#                break
#        else:
#            break
        df_2=df_1.copy()
        df_2['delta_x']=df_2['x'].diff().shift(1).fillna(df_2['x'].diff().shift(-1))
        df_2['delta_y']=df_2['y'].diff().shift(1).fillna(df_2['y'].diff().shift(-1))
        df_2=df_2.loc[df_2['delta_y']>1]
        df_2=df_2.append(df_1.iloc[0:1,:])
        print(df_2)
        if len(df_2.index.tolist())>1 and difference_boo==True:
            break
    if len(df_2.index.tolist())>0:
        w,h=info['width'].tolist()[0],info['height'].tolist()[0]
        img_dup=img.copy()
        for index,row in df_2.iterrows():
            cv2.rectangle(img_dup,(int(row['x']),int(row['y'])),(int(row['x']+w),int(row['y']+h)),[155,155,155],2)
        
        cv2.imshow('thing',img_dup)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return df_2
    
for val in range(20):
    df,img=init_img_filter('source/scores/bartok_piano/img_2.png')
    df1=find_g_clefs(df,img)
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    