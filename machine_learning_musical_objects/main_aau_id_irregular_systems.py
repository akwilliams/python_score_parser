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
    print(df)
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

    '''
    score_data,img_2=find_stave_data('source/scores/img_20.png',init_filter_thresh=190,width_thresh=0.2,delta_y_thresh=0.1252)
    
    cv2.imshow('img',img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

def find_stave_data(img,bounds_thresh=0.5,bounds_size_ratio=0.01,init_filter_thresh=45,width_ratio_std=0.8,width_thresh=0.2,delta_y_thresh=0.4):

    img_info,temp_img=find_score_bounds(img,bounds_thresh,bounds_size_ratio)
    print(img_info)
    width_ratio=width_ratio_std*img_info['width'][np.where(img_info['id']=='score_bounds')[0][0]]/img_info['width'][np.where(img_info['id']=='img_bounds')[0][0]]
    df,temp_img_0=parse_staves(img,init_filter_thresh,width_ratio)
    df=drop_by_perc_off_mean(df.copy(),'width','percent_off_mean_width',width_thresh) 
    df=df.reset_index(drop=True)
    df['delta_y']=df.center_y.diff().shift(-1).fillna(0)
    df_delta_y=df.groupby(['delta_y']).size().reset_index(name='count')
    mean=df_delta_y['delta_y'].mean()-df_delta_y['delta_y'].mode()
    df_delta_y['over_mean']=df_delta_y.loc[df_delta_y['delta_y']<mean,['count']].sum(axis=1)
    df_delta_y['under_mean']=df_delta_y.loc[df_delta_y['delta_y']>mean,['count']].sum(axis=1)
    systems=len(img_info.index[np.where(img_info['id']=='score_bounds')])
    score={'staves_in_system':np.around(((df_delta_y['over_mean'].sum(axis=0)-(systems-1))/systems)+1),'system_count':systems}
    df_2=df.sort_values(['delta_y'],ascending=True)
    '''
    This is a problem that I need to solve
    '''
    df_2=df_2[:int(score['staves_in_system']*score['system_count']-1)]
    df_3=df.loc[np.add(df_2.index.tolist(),[1]*int(score['staves_in_system']*score['system_count']-1))]
    df_2,df_3=df_2.append(df.loc[df['center_y'].idxmin()]),df_3.append(df.loc[[0]])
    df_3,df_2=df_3.sort_values(['center_y'],ascending=True),df_2.sort_values(['center_y'],ascending=True)
    print(df_3,df_2)
    df_3,df_2=df_3.loc[np.mod(df_3.index.tolist(),[5]*len(df_3.index.tolist()))==0],df_2.loc[np.mod(np.add(df_2.index.tolist(),[1]*len(df_2.index.tolist())),[5]*len(df_2.index.tolist()))==0]
    score['staves']=pd.DataFrame(data={'upper_bounds':df_2['center_y'].tolist(),'lower_bounds':df_3['center_y'].tolist(),'left_bounds':df_3['center_x'].subtract(df_3['width'].divide([2]*len(df_3['width'].tolist()))),'right_bounds':df_3['center_x'].add(df_3['width'].divide([2]*len(df_3['width'].tolist())))})
    #if len(score['staves'].index.tolist()) != (score['staves_in_system']*score['system_count']):
    temp=calc_irregular_stave_locations(score,img_info)
    print(temp['stave_count'])
    score['staves_in_system']=temp['stave_count'][:-1].tolist()
    print(temp['stave_count'])
    return score,temp_img_0

def calc_irregular_stave_locations(score_info,img_info):
    
    img_info['upper_bounds']=img_info['center_y'].subtract(img_info['height'].divide([2]*len(img_info.index.tolist())))
    img_info['lower_bounds']=img_info['center_y'].add(img_info['height'].divide([2]*len(img_info.index.tolist())))
    print(score_info['staves'])
    df_temp=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'index':[]})
    for index,row in score_info['staves'].iterrows():
        print(index)
        df_temp=df_temp.append(pd.DataFrame(data={'index':index,'delta_y_upper':img_info.loc[img_info['id']=='score_bounds']['upper_bounds'].subtract([row['upper_bounds']]*len(img_info.loc[img_info['id']=='score_bounds'].index.tolist())),'delta_y_lower':img_info.loc[img_info['id']=='score_bounds']['lower_bounds'].subtract([row['lower_bounds']]*len(img_info.loc[img_info['id']=='score_bounds'].index.tolist()))}))

    print(df_temp)
    df_temp=df_temp.loc[df_temp['delta_y_lower']>0]
    df_temp=df_temp.loc[df_temp['delta_y_upper']<0]
    df_temp['numrow']=df_temp.index.tolist()
    print(df_temp)
    img_info['stave_count']=df_temp['numrow'].value_counts()

    return img_info

def find_clef_data(img,stave_data):
    return stave_data,img


#bx,img=draw_boxes_by_params('source/scores/img_10.png',gBlur_x=11,gBlur_y=11,gBlur_std_dev=9,thresh=125,bx_area_thresh=[-1,-1],bx_angle_thresh=[-1,-1],draw_rects=False):


#score_data,img_2=find_stave_data('source/scores/img_9.png',init_filter_thresh=190,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#
#height=np.abs(score_data['staves']['lower_bounds'].tolist()[0]-score_data['staves']['upper_bounds'].tolist()[0])
#height=int(height if height%2==1 else height+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_9.png',17,71,0,135,[height*(height/3),height*(height*1.5)],[-1,-1],True)
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
#df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#df=df.loc[df['width']<(height*1.75)]
#df=df.loc[df['width']>(height*2/5)]
#df=df.loc[df['height']<(height*1.75)]
#df=df.loc[df['center_x']<600]
##print(len(bx))
##cv2.imshow('img',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#    
#df_5=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df.iterrows():
#    df_5=df_5.append(pd.DataFrame(data={'index':index,'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds'])}))
#    #score_data['staves']['left_bounds'].subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()))
#    #np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds'])
#    #'delta_x_upper':score_data['staves']['right_bounds'].subtract([row['center_x']]*len(score_data['staves']['right_bounds'].tolist()))
#print(df_5)
#df_6=df_5.copy()
#
#df_5=df_5.loc[df_5['delta_y_lower']>0]
#df_5=df_5.loc[df_5['delta_y_upper']<0]
#print(df_5)



#
#
#
#score_data,img_2=find_stave_data('source/scores/img_8.png',init_filter_thresh=140,width_thresh=0.2,delta_y_thresh=0.2)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_8.png',11,55,0,135,[height*(height*(1/7)),height*(height*1.25)],[-1,-1],True)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
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
#
#df=df_3.copy().sort_values(by=['delta_x_lower'])
#df['rownum']=df.index.tolist()
#
#df=df.drop_duplicates(subset='rownum',keep='first')
#
#stave_data = []
#
#
#for index in range(int(score_data['staves_in_system'])):
#    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
#    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
#    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
#    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
#    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
#    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
#    
#    stave_data.append(temp.copy())
#    del temp
#
#
#
#
#
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_10.png',init_filter_thresh=185,width_thresh=0.2,delta_y_thresh=0.2)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_10.png',35,101,0,155,[height*(height*2/7),height*(height*1.5)],[-1,-1],True)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
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
#
#df=df_3.copy().sort_values(by=['delta_x_lower'])
#df['rownum']=df.index.tolist()
#
#df=df.drop_duplicates(subset='rownum',keep='first')
#
#stave_data = []
#
#
#for index in range(int(score_data['staves_in_system'])):
#    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
#    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
#    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
#    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
#    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
#    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
#    
#    stave_data.append(temp.copy())
#    del temp
#
#
#'''
#As long as there is no perc off mean larger than 0.3 then it should be indicative of a stave with a similar clef
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
#score_data,img_2=find_stave_data('source/scores/img_11.png',init_filter_thresh=155,width_thresh=0.2,delta_y_thresh=0.2)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_11.png',33,101,0,155,[height*(height*(2/7)),height*(height*1.5)],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
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
#
#df=df_3.copy().sort_values(by=['delta_x_lower'])
#df['rownum']=df.index.tolist()
#
#df=df.drop_duplicates(subset='rownum',keep='first')
#
#stave_data = []
#
#
#for index in range(int(score_data['staves_in_system'])):
#    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
#    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
#    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
#    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
#    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
#    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
#    
#    stave_data.append(temp.copy())
#    del temp
#
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_12.png',init_filter_thresh=125,width_thresh=0.2,delta_y_thresh=0.2)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_12.png',33,101,0,155,[height*(height*(2/7)),height*(height*1.5)],[-1,-1],True)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
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
#
#df=df_3.copy().sort_values(by=['delta_x_lower'])
#df['rownum']=df.index.tolist()
#
#df=df.drop_duplicates(subset='rownum',keep='first')
#
#stave_data = []
#
#
#for index in range(int(score_data['staves_in_system'])):
#    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
#    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
#    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
#    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
#    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
#    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
#    
#    stave_data.append(temp.copy())
#    del temp
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_13.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)
#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
#height=int(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#bx,img=draw_boxes_by_params('source/scores/img_13.png',33,101,0,155,[height*(height*4/7),height*(height*3/2)],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
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
#
#df=df_3.copy().sort_values(by=['delta_x_lower'])
#df['rownum']=df.index.tolist()
#
#df=df.drop_duplicates(subset='rownum',keep='first')
#
#stave_data = []
#
#
#for index in range(int(score_data['staves_in_system'])):
#    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
#    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
#    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
#    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
#    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
#    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
#    
#    stave_data.append(temp.copy())
#    del temp




score_data,img_2=find_stave_data('source/scores/img_14.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)

cv2.imshow('img',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

score_data['staves']=score_data['staves'].reset_index(drop=True)

score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)



thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
height=int(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
bx,img=draw_boxes_by_params('source/scores/img_14.png',33,101,0,155,[height*(height*2/7),height*(height*3/2)],[-1,-1],True)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
df_2 = pd.concat([df_0,df_1],axis=1)

df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
for index,row in df_2.iterrows():
    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))

df_3=df_3.loc[df_3['delta_y_lower']>0]
df_3=df_3.loc[df_3['delta_y_upper']<0]
df_3=df_3.loc[df_3['ratio']<10]

df=df_3.copy().sort_values(by=['delta_x_lower'])
df['rownum']=df.index.tolist()

df=df.drop_duplicates(subset='rownum',keep='first')

stave_data = []


for index in range(int(score_data['staves_in_system'])):
    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
    
    stave_data.append(temp.copy())
    del temp


score_data,img_2=find_stave_data('source/scores/img_15.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)

cv2.imshow('img',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

score_data['staves']=score_data['staves'].reset_index(drop=True)

score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)



thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
height=int(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
bx,img=draw_boxes_by_params('source/scores/img_15.png',15,73,0,155,[height*(height*2/7),height*(height*3/2)],[-1,-1],True)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
df_2 = pd.concat([df_0,df_1],axis=1)

df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
for index,row in df_2.iterrows():
    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))

df_3=df_3.loc[df_3['delta_y_lower']>0]
df_3=df_3.loc[df_3['delta_y_upper']<0]
df_3=df_3.loc[df_3['ratio']<10]

df=df_3.copy().sort_values(by=['delta_x_lower'])
df['rownum']=df.index.tolist()

df=df.drop_duplicates(subset='rownum',keep='first')

stave_data = []


for index in range(int(score_data['staves_in_system'])):
    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
    
    stave_data.append(temp.copy())
    del temp


score_data,img_2=find_stave_data('source/scores/img_16.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)

cv2.imshow('img',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

score_data['staves']=score_data['staves'].reset_index(drop=True)

score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)



thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
height=int(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
bx,img=draw_boxes_by_params('source/scores/img_16.png',15,73,0,155,[height*(height*2/7),height*(height*3/2)],[-1,-1],True)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
df_2 = pd.concat([df_0,df_1],axis=1)

df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
for index,row in df_2.iterrows():
    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))

df_3=df_3.loc[df_3['delta_y_lower']>0]
df_3=df_3.loc[df_3['delta_y_upper']<0]
df_3=df_3.loc[df_3['ratio']<10]

df=df_3.copy().sort_values(by=['delta_x_lower'])
df['rownum']=df.index.tolist()

df=df.drop_duplicates(subset='rownum',keep='first')

stave_data = []


for index in range(int(score_data['staves_in_system'])):
    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
    
    stave_data.append(temp.copy())
    del temp





score_data,img_2=find_stave_data('source/scores/img_16.png',init_filter_thresh=155,width_thresh=0.12,delta_y_thresh=0.1252)

cv2.imshow('img',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

score_data['staves']=score_data['staves'].reset_index(drop=True)

score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)



thing=score_data['staves']['lower_bounds'].subtract(score_data['staves']['upper_bounds'])
height=int(thing.mode()[0] if thing.mode()[0]%2==1 else thing.mode()[0]+1)
bx,img=draw_boxes_by_params('source/scores/img_16.png',15,73,0,155,[height*(height*2/7),height*(height*3/2)],[-1,-1],True)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
df_1['height'],df_1['width']=df_1.min(axis=1),df_1.max(axis=1)
df_2 = pd.concat([df_0,df_1],axis=1)

df_3=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
for index,row in df_2.iterrows():
    df_3=df_3.append(pd.DataFrame(data={'index':index,'angle':np.abs(row['angle']),'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))

df_3=df_3.loc[df_3['delta_y_lower']>0]
df_3=df_3.loc[df_3['delta_y_upper']<0]
df_3=df_3.loc[df_3['ratio']<10]

df=df_3.copy().sort_values(by=['delta_x_lower'])
df['rownum']=df.index.tolist()

df=df.drop_duplicates(subset='rownum',keep='first')

stave_data = []


for index in range(int(score_data['staves_in_system'])):
    #stave_data.append(df.loc[df['rownum']%score_data['system_count']==index].copy())
    temp=df.loc[df['rownum']%int(score_data['staves_in_system'])==index].copy()
    #temp=calc_perc_off_mean(temp.copy(),'delta_x_lower','perc_off_mean_delta_x')
    temp=calc_perc_off_mean(temp.copy(),'area','perc_off_mean_area')
    temp=calc_perc_off_mean(temp.copy(),'ratio','perc_off_mean_ratio')
    temp=calc_perc_off_mean(temp.copy(),'angle','perc_off_mean_angle')
    
    stave_data.append(temp.copy())
    del temp


print(np.mod(stave_data[0].index.tolist(),[5]*len(stave_data[0].index.tolist())))
print(stave_data[0].loc[np.mod(stave_data[0].index.tolist(),[5]*len(stave_data[0].index.tolist()))==0])

#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
