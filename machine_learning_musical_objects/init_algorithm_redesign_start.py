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
    hold,img=draw_boxes_by_params(stave_temp,filter_arg,1,0,thresh,[-1,-1],[-1,-1],True)
    df_0,df_1 = write_bndng_bx_pd_df([hold],'stave_line')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df = pd.concat([df_0,df_1],axis=1)
    return df,img

def find_score_bounds(img,thresh=0.5,size_ratio=0.1):
    temp_img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    height,width=temp_img.shape[:2]
    hold,score_temp=draw_boxes_by_params(temp_img,1,1,300,245,[(height*width*size_ratio),(height*width)],[-1,-1],True)
    temp_df=pd.DataFrame([[0,width/2,height/2,'img_bounds',height,width,(height*width)]],columns=['angle','center_x','center_y','id','height','width','area'])
    df,df_1=write_bndng_bx_pd_df([hold],'score_bounds')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df_1['area']=df_1.apply(lambda row: row.height*row.width, axis=1)
    df=pd.concat([df,df_1],axis=1)
    df=df.drop(df.index[np.where((df['area']/df['area'].max()<thresh))[0]])
    df=df.sort_values(by=['center_y'],ascending=True)
    df=df.append(temp_df)
    df=df.reset_index(drop=True)
    return df,score_temp

def calc_perc_off_mean(df,index,destination):
    mean=df[index].mean()
    df[destination]=np.abs((df[index]-mean)/mean)
    return df

def drop_by_perc_off_mean(df,ident,destination,threshold=1,counter=1,mode=0):
    df=calc_perc_off_mean(df,ident,destination)
    if mode==0:
        while df[destination].max()>threshold:
            df=df.drop(df[destination].idxmax())
            df=calc_perc_off_mean(df,ident,destination)
    elif mode==1:
        while len(df[destination].index.tolist())>counter:
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
    df=df.reset_index(drop=True)
    df['delta_y']=df.center_y.diff().shift(-1).fillna(0)
    df_delta_y=df.groupby(['delta_y']).size().reset_index(name='count')
    mean=df_delta_y['delta_y'].mean()-df_delta_y['delta_y'].mode()
    df_delta_y['over_mean']=df_delta_y.loc[df_delta_y['delta_y']<mean,['count']].sum(axis=1)
    df_delta_y['under_mean']=df_delta_y.loc[df_delta_y['delta_y']>mean,['count']].sum(axis=1)
    df_2=df.sort_values(['delta_y'],ascending=True)
    score={'total_staves':int(len(df_2.loc[df_2['delta_y']<(1.3*df_2['delta_y'].mode()[0])])+1),'system_count':len(img_info.index[np.where(img_info['id']=='score_bounds')])}
    df_2=df_2[:(score['total_staves']-1)]
    df_3=df.loc[np.add(df_2.index.tolist(),([1]*(score['total_staves']-1)))]
    df_2,df_3=df_2.append(df.loc[df['center_y'].idxmin()]),df_3.append(df.loc[[0]])
    df_3,df_2=df_3.sort_values(['center_y'],ascending=True),df_2.sort_values(['center_y'],ascending=True)
    df_3['mod_index'],df_2['mod_index']=np.mod(df_3.index.tolist(),[5]*len(df_2.index.tolist())),np.mod(df_2.index.tolist(),[5]*len(df_2.index.tolist()))
    df_3['numrow'],df_2['numrow']=df_3.index.tolist(),df_2.index.tolist()
    df_3['mod_index']=[(df_3['mod_index'].tolist()[i]-5) if df_3['mod_index'].tolist()[i]>df_3.loc[df_3['numrow']==df_3['numrow'].min()]['mod_index'][0] else df_3['mod_index'].tolist()[i] for i in range(len(df_3['mod_index'].tolist()))]
    df_2['mod_index']=df_2['mod_index'].subtract([df_2.loc[df_2['numrow'].idxmin()]['mod_index']]*len(df_2.index.tolist()))
    df_3['numrow'],df_2['numrow']=df_3['numrow'].subtract(df_3['mod_index']),df_2['numrow'].subtract(df_2['mod_index'])
    
    df_3,df_2=df_3.loc[np.mod(df_3['numrow'].tolist(),[5]*len(df_3['numrow'].tolist()))==0],df_2.loc[np.mod(np.add(df_2['numrow'].tolist(),[1]*len(df_2['numrow'].tolist())),[5]*len(df_2['numrow'].tolist()))==0]
 
    score['staves']=pd.DataFrame(data={'upper_bounds':df_2['center_y'].tolist(),'lower_bounds':df_3['center_y'].tolist(),'left_bounds':df_3['center_x'].subtract(df_3['width'].divide([2]*len(df_3['width'].tolist()))),'right_bounds':df_3['center_x'].add(df_3['width'].divide([2]*len(df_3['width'].tolist())))})
    
    img_info['upper_bounds']=img_info['center_y'].subtract(img_info['height'].divide([2]*len(img_info.index.tolist())))
    img_info['lower_bounds']=img_info['center_y'].add(img_info['height'].divide([2]*len(img_info.index.tolist())))
    df_temp=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'index':[]})

    for index,row in score['staves'].iterrows():
        df_temp=df_temp.append(pd.DataFrame(data={'index':index,'delta_y_upper':img_info.loc[img_info['id']=='score_bounds']['upper_bounds'].subtract([row['upper_bounds']]*len(img_info.loc[img_info['id']=='score_bounds'].index.tolist())),'delta_y_lower':img_info.loc[img_info['id']=='score_bounds']['lower_bounds'].subtract([row['lower_bounds']]*len(img_info.loc[img_info['id']=='score_bounds'].index.tolist()))}))

    df_temp=df_temp.loc[df_temp['delta_y_lower']>0]
    df_temp=df_temp.loc[df_temp['delta_y_upper']<0]
    df_temp['numrow']=df_temp.index.tolist()
    img_info['stave_count']=df_temp['numrow'].value_counts()
    score['staves_in_system']=img_info['stave_count'][:-1].tolist()

    score['staves']=score['staves'].reset_index(drop=True)
    score['staves']['delta_upper_bounds']=score['staves'].upper_bounds.diff().shift(-1).fillna(0)
    score['staves']['delta_lower_bounds']=score['staves'].lower_bounds.diff().shift(-1).fillna(0)

    hold=score['staves']['lower_bounds'].subtract(score['staves']['upper_bounds'])
    height=int(hold.mode()[0] if hold.mode()[0]%2==1 else hold.mode()[0]+1)
    divisor=score['staves']['upper_bounds'].diff().shift(1).mean()
    bxs,score_img=draw_boxes_by_params(img,1,height,0,145,[height*9,height**3],[-1,-1],False)
    df_4,df_5=write_bndng_bx_pd_df([bxs],'group_pass')
    df_5['height'],df_5['width']=df_5.max(axis=1),df_5.min(axis=1)
    df_6 = pd.concat([df_4,df_5],axis=1)
    df_6['ratio']=df_6['height'].divide(df_6['width'])
    df_6=df_6.loc[df_6['ratio']>35]
    df_6['staves_in_group']=df_6['height'].divide(divisor).round(0)
    df_6=df_6.loc[df_6['staves_in_group']<=np.amax(score['staves_in_system'])]
    df_6=df_6.sort_values(by=['center_y'],ascending=True)
    df_6=df_6.reset_index(drop=True)
    df_6['delta_y']=df_6.center_y.diff().shift(-1).fillna(height+1)
    df_6['numrow']=df_6.index.tolist()
    df_7=df_6.loc[df_6['delta_y']>height]
    df_7['delta']=df_7['numrow'].diff().shift(-1).fillna(df_7.loc[df_7['delta_y']>10]['numrow'].diff().shift(1))
    container={'voice_count':[],'center_y':[]}

    for index,row in df_7.iterrows():
        if index-1 not in (df_7.index.tolist()+[-1]):
            container['voice_count'].append(row['staves_in_group']+1)
            container['center_y'].append(row['center_y'])

        score['stave_groups']=pd.DataFrame(data=container)

    return score,temp_img_0

def find_g_clefs(img,stave_data,x_blur=15,y_blur=73,init_filter=155,exclude_staves=[]):
    
    height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
    height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
    bx,img=draw_boxes_by_params(img,x_blur,y_blur,0,init_filter,[height*(height/3),height*(height*7/4)],[-1,-1],True)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
    df_2 = pd.concat([df_0,df_1],axis=1)
    
    df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
    for index,row in df_2.iterrows():
        df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))

    df_3=df_3.loc[df_3['delta_y_lower']>0]
    df_3=df_3.loc[df_3['delta_y_upper']<0]
    df_3=df_3.loc[df_3['ratio']<10]
    df_3=df_3.loc[df_3['angle']>0.3]
    df_3['numrow']=df_3.index.tolist()
    df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
    df_3=df_3.drop_duplicates(subset='numrow',keep='first')
    print(df_3)
    for value in exclude_staves:
        df_3=df_3.loc[df_3['numrow']!=value]
    df_3=df_3.loc[df_3['ratio']>1.81]
    df_3=df_3.sort_values(by=['area'],ascending=True)
    df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
    df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
    df_3=df_3.reset_index(drop=True)
    print(df_3)
    if df_3['perc_delta_of_area'].max()>0.15:
        df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
    df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
    print(df_3)
    df_3['distance_to_mean_sum']=np.abs(df_3['sum'].subtract([df_3['sum'].mean()]*len(df_3.index.tolist())))
    print(df_3)
    df_3=df_3.loc[df_3['distance_to_mean_sum']<20]
    
    clef_data=df_3['numrow'].tolist()
    
    return clef_data

#def find_clef(img,stave_data,x_blur=15,y_blur=73,init_filter=155,exclude_staves=[],kind='g_clef',return_img=False,parameters={'use_preset':True}):
#    
#    #Lines 266-280 find objects via filter and filter all but the first on the stave
#    height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#    height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#    bx,img=draw_boxes_by_params(img,x_blur,y_blur,0,init_filter,[height*(height/3),height*(height*7/4)],[-1,-1],True)
#
#    df_0,df_1=write_bndng_bx_pd_df([bx],(kind+' pass'))
#    df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#    df_2 = pd.concat([df_0,df_1],axis=1)
#    
#    df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#    for index,row in df_2.iterrows():
#        df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#    df_3=df_3.loc[df_3['delta_y_lower']>0]
#    df_3=df_3.loc[df_3['delta_y_upper']<0]
#    
#    if parameters['use_preset'] == True:#Here is where I can use machine learning algorithms to tighten up the preset parameter values
#        if kind == 'g_clef':
#            parameters={}
#        elif kind == 'c_clef':
#            parameters={}
#        elif kind == 'percussion':
#            parameters={}
#        elif kind == 'f_clef':
#            parameters={}
#    
#    #Lines 293-302 filter data by clef data patterns
#    df_3=df_3.loc[df_3['ratio']<parameters['ratio_filter']]
#    df_3=df_3.loc[df_3['angle']>parameters['angle_filter']]
#    df_3['numrow']=df_3.index.tolist()
#    df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#    df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#    for value in exclude_staves:
#        df_3=df_3.loc[df_3['numrow']!=value]
#    #Data up to this point would be the data set for the machine learning algorithm (because every thing below it has to do with filtering based on desired clef types)
#    #Also after training the machine learning algorithm, I will not need the 
#    df_3=df_3.loc[df_3['ratio']>parameters['ratio_min']]
#    df_3=df_3.loc[df_3['ratio']<parameters['ratio_max']]
#    
#    
#
#'''
#
#'''
#
#
#
#
#
#
#
#
#
#clef_data={'g_clef':[],'f_clef':[],'c_clef':[]}
#score_data,img_2=find_stave_data('source/scores/img_6.png',init_filter_thresh=180,width_thresh=0.12,delta_y_thresh=0.12)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data,img_2=find_stave_data('source/scores/img_8.png',init_filter_thresh=140,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data,img_2=find_stave_data('source/scores/img_9.png',init_filter_thresh=170,width_thresh=0.2,delta_y_thresh=0.2)
#'''
#score_data,img_2=find_stave_data('source/scores/img_10.png',init_filter_thresh=135,width_thresh=0.082,delta_y_thresh=0.2)
#'''
#score_data,img_2=find_stave_data('source/scores/img_11.png',init_filter_thresh=155,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data,img_2=find_stave_data('source/scores/img_12.png',init_filter_thresh=125,width_thresh=0.2,delta_y_thresh=0.2)
#'''
#score_data,img_2=find_stave_data('source/scores/img_13.png',init_filter_thresh=205,width_thresh=0.12,delta_y_thresh=0.252)
#'''
#score_data,img_2=find_stave_data('source/scores/img_14.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_14.png',score_data,9,43,125)
#
#score_data,img_2=find_stave_data('source/scores/img_15.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#
#score_data,img_2=find_stave_data('source/scores/img_16.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#
#score_data,img_2=find_stave_data('source/scores/img_17.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#
#score_data,img_2=find_stave_data('source/scores/img_18.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#
#score_data,img_2=find_stave_data('source/scores/img_19.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#
#score_data,img_2=find_stave_data('source/scores/img_20.png',init_filter_thresh=190,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_20.png',score_data,15,73,125)
##clef_data['c_clef']=find_c_clefs('source/scores/img_20.png',score_data,15,95,125)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_20.png',1,15,0,25,[0,5000],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<10]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#print(df_3['numrow'].tolist())
#
#
#
#
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_21.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_21.png',score_data,15,73,135)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_21.png',49,75,0,125,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<10]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#print(df_3['numrow'].tolist())
#
#
#
#
#
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_22.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_22.png',score_data,15,73,155)
##Only G_Clefs
#
#'''
#score_data,img_2=find_stave_data('source/scores/img_23.png',init_filter_thresh=190,width_thresh=0.12,delta_y_thresh=0.1252)
#'''
#score_data,img_2=find_stave_data('source/scores/img_24.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_24.png',score_data,15,73,155)
##Only G_Clefs
#
#score_data,img_2=find_stave_data('source/scores/img_25.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_25.png',score_data,15,73,155)
##No C_clefs
#
#score_data,img_2=find_stave_data('source/scores/img_26.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_26.png',score_data,21,73,178)
#
#score_data,img_2=find_stave_data('source/scores/img_27.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_27.png',score_data,15,73,145)
#
#score_data,img_2=find_stave_data('source/scores/img_28.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_28.png',score_data,15,73,105)
#clef_data=clef_data+find_g_clefs('source/scores/img_28.png',score_data,15,73,105,exclude_staves=clef_data)
#
#score_data,img_2=find_stave_data('source/scores/img_29.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_29.png',score_data,15,73,165)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_29.png',15,45,0,166,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<10]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#print(df_3['numrow'].tolist())
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_30.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_30.png',score_data,15,73,155)
#
#score_data,img_2=find_stave_data('source/scores/img_31.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_31.png',score_data,21,73,135)
#
#score_data,img_2=find_stave_data('source/scores/img_32.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_32.png',score_data,21,73,155)
#
#score_data,img_2=find_stave_data('source/scores/img_33.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_33.png',score_data,21,73,155)
#
#score_data,img_2=find_stave_data('source/scores/img_34.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_34.png',score_data,15,73,155)
#
#score_data,img_2=find_stave_data('source/scores/img_35.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_35.png',score_data,15,73,155)
#'''
#score_data,img_2=find_stave_data('source/scores/img_36.png',init_filter_thresh=125,width_thresh=0.1,delta_y_thresh=0.1)
#clef_data=find_g_clefs('source/scores/img_36.png',score_data,11,73,175)
#'''
#score_data,img_2=find_stave_data('source/scores/img_37.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#'''
#clef_data=find_g_clefs('source/scores/img_37.png',score_data,21,73,185)
#'''
#'''
#score_data,img_2=find_stave_data('source/scores/img_38.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_38.png',score_data,15,73,105)
#'''
#score_data,img_2=find_stave_data('source/scores/img_39.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_39.png',score_data,15,73,165)
#
#score_data,img_2=find_stave_data('source/scores/img_40.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_40.png',score_data,15,73,175)
#'''
#score_data,img_2=find_stave_data('source/scores/img_41.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_41.png',score_data,21,73,135)
#'''
#score_data,img_2=find_stave_data('source/scores/img_42.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_42.png',score_data,21,73,175)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_42.png',15,65,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_43.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_43.png',score_data,21,73,175)
#
#score_data,img_2=find_stave_data('source/scores/img_44.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_44.png',score_data,15,73,195)
#'''
#score_data,img_2=find_stave_data('source/scores/img_45.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_45.png',score_data,21,73,165)
#'''
#score_data,img_2=find_stave_data('source/scores/img_46.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_46.png',score_data,15,73,178)
#
#score_data,img_2=find_stave_data('source/scores/img_47.png',init_filter_thresh=185,width_thresh=0.12,delta_y_thresh=0.1252)
#'''
#clef_data=find_g_clefs('source/scores/img_47.png',score_data,15,73,185)
#'''
#score_data,img_2=find_stave_data('source/scores/img_48.png',init_filter_thresh=145,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_48.png',score_data,15,73,175)
#
#score_data,img_2=find_stave_data('source/scores/img_49.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data=find_g_clefs('source/scores/img_49.png',score_data,15,73,165)
#'''
#clef_data=clef_data+find_g_clefs('source/scores/img_49.png',score_data,21,73,180,exclude_staves=clef_data)
#'''
#score_data,img_2=find_stave_data('source/scores/img_50.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#'''
#clef_data=find_g_clefs('source/scores/img_50.png',score_data,11,73,95)
#'''
#
#score_data,img_2=find_stave_data('source/scores/img_51.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_51.png',score_data,21,73,175)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_51.png',15,65,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_52.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_52.png',score_data,21,73,175)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_52.png',15,65,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_53.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_53.png',score_data,21,73,175)
#
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_53.png',21,73,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())
#
#
#score_data,img_2=find_stave_data('source/scores/img_54.png',init_filter_thresh=175,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_54.png',score_data,21,73,175)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_54.png',21,73,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())
#
#
#score_data,img_2=find_stave_data('source/scores/img_55.png',init_filter_thresh=165,width_thresh=0.12,delta_y_thresh=0.1252)
#clef_data['g_clef']=find_g_clefs('source/scores/img_55.png',score_data,21,73,175)
#
#height_calc=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(height_calc.mode()[0] if height_calc.mode()[0]%2==1 else height_calc.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_55.png',7,21,0,175,[height*(height*2/7),height*(height*7/4)],[-1,-1],True)
#'''
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#'''
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass_g_c')
#df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
#df_2 = pd.concat([df_0,df_1],axis=1)
#    
#df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df_2.iterrows():
#    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#df_3=df_3.loc[df_3['delta_y_lower']>0]
#df_3=df_3.loc[df_3['delta_y_upper']<0]
#df_3=df_3.loc[df_3['ratio']<5]
#df_3=df_3.loc[df_3['angle']>0.3]
#df_3['numrow']=df_3.index.tolist()
#df_3=df_3.sort_values(by=['delta_x_lower'],ascending=True)
#df_3=df_3.drop_duplicates(subset='numrow',keep='first')
#print(df_3)
#for value in clef_data['g_clef']:
#    df_3=df_3.loc[df_3['numrow']!=value]
#print(df_3)
#df_3=df_3.loc[df_3['ratio']<1.79]
#df_3=df_3.loc[df_3['ratio']>1.2]
#print(df_3)
#df_3=df_3.sort_values(by=['area'],ascending=True)
#df_3['delta_area']=df_3['area'].diff().shift(-1).fillna(df_3['area'].diff().shift(1))
#df_3['perc_delta_of_area']=df_3['delta_area'].divide(df_3['area'])
#df_3=df_3.reset_index(drop=True)
#print(df_3)
#if df_3['perc_delta_of_area'].max()>0.15:
#    df_3=df_3.iloc[(df_3['perc_delta_of_area'].idxmax()+1): :]
#df_3=drop_by_perc_off_mean(df_3.copy(),'area','perc_off_mean_area',0.3)
#print(df_3)
#
#    
#print(df_3['numrow'].tolist())

