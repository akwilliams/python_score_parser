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
    systems=len(img_info.index[np.where(img_info['id']=='score_bounds')])
    score={'staves_in_system':np.around(((df_delta_y['over_mean'].sum(axis=0)-(systems-1))/systems)+1),'system_count':systems}
    df_2=df.sort_values(['delta_y'],ascending=True)
    df_2=df_2[:int(score['staves_in_system']*score['system_count']-1)]
    df_3=df.loc[np.add(df_2.index.tolist(),[1]*int(score['staves_in_system']*score['system_count']-1))]
    df_2,df_3=df_2.append(df.loc[df['center_y'].idxmin()]),df_3.append(df.loc[[0]])
    df_3,df_2=df_3.sort_values(['center_y'],ascending=True),df_2.sort_values(['center_y'],ascending=True)
    score['staves']=pd.DataFrame(data={'upper_bounds':df_2['center_y'].tolist(),'lower_bounds':df_3['center_y'].tolist(),'left_bounds':df_3['center_x'].subtract(df_3['width'].divide([2]*len(df_3['width'].tolist()))),'right_bounds':df_3['center_x'].add(df_3['width'].divide([2]*len(df_3['width'].tolist())))})
    return score,temp_img_0

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






#score_data,img_2=find_stave_data('source/scores/img_8.png',init_filter_thresh=140,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#
#
#height=np.abs(score_data['staves']['lower_bounds'].tolist()[0]-score_data['staves']['upper_bounds'].tolist()[0])
#height=int(height if height%2==1 else height+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_8.png',17,71,0,155,[height*(height*(2/7)),height*(height*1.5)],[-1,-1],True)
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
#df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#df=df.loc[df['width']<(height*2.5)]
#df=df.loc[df['width']>(height*2/5)]
#df=df.loc[df['height']<(height*2.5)]
#df=df.loc[df['center_x']<500]
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
##df_5=df_5.loc[df_5['delta_y_lower']+df_5['delta_y_upper']<height*0.7]
#print(df_5)









#score_data,img_2=find_stave_data('source/scores/img_10.png',init_filter_thresh=190,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)
#
#   
#height=np.abs(score_data['staves']['lower_bounds'].tolist()[0]-score_data['staves']['upper_bounds'].tolist()[0])
#height=int(height if height%2==1 else height+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_10.png',35,101,0,155,[height*(height/3),height*(height*1.5)],[-1,-1],True)
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
#df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#df=df.loc[df['width']<(height*7/3)]
#df=df.loc[df['width']>(height*2/5)]
#df=df.loc[df['height']<(height*2.5)]
#df=df.loc[df['center_x']<750]
#
#df_5=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df.iterrows():
#    df_5=df_5.append(pd.DataFrame(data={'index':index,'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#    #score_data['staves']['left_bounds'].subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()))
#    #np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds'])
#    #'delta_x_upper':score_data['staves']['right_bounds'].subtract([row['center_x']]*len(score_data['staves']['right_bounds'].tolist()))
#
#df_6=df_5.copy()
#
#df_5=df_5.loc[df_5['delta_y_lower']>0]
#df_5=df_5.loc[df_5['delta_y_upper']<0]
#
#df_5=df_5.loc[df_5['delta_y_lower']<height]
#df_5=df_5.loc[df_5['delta_y_upper']>-1*height]
#
#print(df_5)
#df_6=df_5.loc[df_5.index.duplicated(keep=False)]
#print(df_5.loc[df_5.index.duplicated(keep=False)])
#print(df_6['delta_x_lower'].max())
#df_5=df_5[df_5['delta_x_lower'] != df_6['delta_x_lower'].max()]
#print(df_5)
#
#df_5=calc_perc_off_mean(df_5.copy(),'area','perc_off_mean_area')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_lower','perc_off_mean_yl')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_upper','perc_off_mean_yu')
#df_5=calc_perc_off_mean(df_5.copy(),'ratio','perc_off_mean_ratio')
#df_5=calc_perc_off_mean(df_5.copy(),'sum','perc_off_mean_sum')
#
#
#df_5['perc_sum']=df_5['perc_off_mean_area'].add(df_5['perc_off_mean_yl'].add(df_5['perc_off_mean_yu'].add(df_5['perc_off_mean_ratio'].add(df_5['perc_off_mean_sum'])))).divide([5]*len(df_5.index.tolist()))
#
#
#print(df_5)
#df_5=df_5.drop(df_5['perc_sum'].idxmax())
#print(df_5)
#
#df_5=calc_perc_off_mean(df_5.copy(),'area','perc_off_mean_area')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_lower','perc_off_mean_yl')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_upper','perc_off_mean_yu')
#df_5=calc_perc_off_mean(df_5.copy(),'ratio','perc_off_mean_ratio')
#df_5=calc_perc_off_mean(df_5.copy(),'sum','perc_off_mean_sum')
#
#df_5['perc_sum']=df_5['perc_off_mean_area'].add(df_5['perc_off_mean_yl'].add(df_5['perc_off_mean_yu'].add(df_5['perc_off_mean_ratio'].add(df_5['perc_off_mean_sum'])))).divide([5]*len(df_5.index.tolist()))
#print(df_5)
#
#
#
#print(df_5['delta_y_lower'].mean(),df_5['delta_y_upper'].mean(),df_5['delta_x_lower'].mean())
##df_6['dyuom']=np.subtract(([df_5['delta_y_upper'].mean()]*(len(df_6.index.tolist()))),df_6['delta_y_upper'])
#df_6['dyuom']=df_6['delta_y_upper'].subtract(([df_5['delta_y_upper'].mean()]*(len(df_6.index.tolist())))).abs()
#df_6['dylom']=df_6['delta_y_lower'].subtract(([df_5['delta_y_lower'].mean()]*(len(df_6.index.tolist())))).abs()
#df_6['dxlom']=df_6['delta_x_lower'].subtract(([df_5['delta_x_lower'].mean()]*(len(df_6.index.tolist())))).abs()
#print(df_6)
#df_9=calc_perc_off_mean(df_5.copy(),'delta_x_lower','p_o_d_x_l')
#df_10=calc_perc_off_mean(df_5.copy(),'delta_y_lower','p_o_d_y_l')
#df_11=calc_perc_off_mean(df_5.copy(),'delta_y_upper','p_o_d_y_u')
#df_9=df_9.loc[df_9['p_o_d_x_l'] > 0.45]
#df_10=df_10.loc[df_10['p_o_d_y_l'] > 0.45]
#df_11=df_11.loc[df_11['p_o_d_y_u'] > 0.45]
#
#
##del df_9['p_o_d_x_l']
##del df_10['p_o_d_y_l']
##del df_11['p_o_d_y_u']
##df_12=df_9
##df_12=df_12.append(df_10)
##df_12=df_12.append(df_11)
##
##df_12=df_12[df_12.duplicated()]
##print(df_12)
##
##df_5=df_5.append(df_12)
##df_5=df_5.drop_duplicates(keep=False)
##print(df_5)
##
##df_13=df_5.drop(df_5['delta_y_upper'].idxmax())
##df_14=df_5.drop(df_5['delta_y_lower'].idxmax())
##
##df_16=df_13.append(df_14)
##print(df_13,df_14)
##
##df_15=df_16[df_16.duplicated()]
##print(df_15)
#
#



#score_data,img_2=find_stave_data('source/scores/img_11.png',init_filter_thresh=190,width_thresh=0.2,delta_y_thresh=0.2)
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
#height=np.abs(score_data['staves']['lower_bounds'].tolist()[0]-score_data['staves']['upper_bounds'].tolist()[0])
#height=int(height if height%2==1 else height+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_11.png',33,101,0,155,[height*(height*(2/7)),height*(height*1.5)],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
#df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#df=df.loc[df['width']<(height*2.5)]
#df=df.loc[df['width']>(height*2/5)]
#df=df.loc[df['height']<(height*2.5)]
#df=df.loc[df['center_x']<500]
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
##df_5=df_5.loc[df_5['delta_y_lower']+df_5['delta_y_upper']<height*0.7]
#print(df_5)
    






#score_data,img_2=find_stave_data('source/scores/img_12.png',init_filter_thresh=190,width_thresh=0.2,delta_y_thresh=0.2)
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
#height=np.abs(score_data['staves']['lower_bounds'].tolist()[0]-score_data['staves']['upper_bounds'].tolist()[0])
#height=int(height if height%2==1 else height+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_12.png',33,101,0,155,[height*(height*(2/7)),height*(height*1.5)],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
#df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#df=df.loc[df['width']<(height*2.5)]
#df=df.loc[df['width']>(height*2/5)]
#df=df.loc[df['height']<(height*2.5)]
#df=df.loc[df['center_x']<500]
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
##df_5=df_5.loc[df_5['delta_y_lower']+df_5['delta_y_upper']<height*0.7]
#print(df_5)
    








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
#height=int(thing if thing.mode()[0]%2==1 else thing.mode()[0]+1)
#print(height)
#bx,img=draw_boxes_by_params('source/scores/img_13.png',33,101,0,155,[height*(height*4/7),height*(height*3/2)],[-1,-1],True)
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#df_0,df_1=write_bndng_bx_pd_df([bx],'clef_pass')
##df_1['height'],df_1['width']=df_1.max(axis=1),df_1.min(axis=1)
#df = pd.concat([df_0,df_1],axis=1)
#
#df_5=pd.DataFrame(data={'delta_y_upper':[],'delta_y_lower':[],'delta_x_lower':[],'index':[]})
#for index,row in df.iterrows():
#    df_5=df_5.append(pd.DataFrame(data={'index':index,'delta_y_lower':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())),'delta_y_upper':score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())),'delta_x_lower':np.subtract([row['center_x']]*len(score_data['staves']['left_bounds'].tolist()),score_data['staves']['left_bounds']),'area':row['height']*row['width'],'ratio':row['width']/row['height'],'sum':score_data['staves']['lower_bounds'].subtract([row['center_y']]*len(score_data['staves']['lower_bounds'].tolist())).add(score_data['staves']['upper_bounds'].subtract([row['center_y']]*len(score_data['staves']['upper_bounds'].tolist())))}))
#
#df_5=df_5.loc[df_5['delta_y_lower']>0]
#df_5=df_5.loc[df_5['delta_y_upper']<0]
#df_5=df_5.reset_index()
#
#
#df_6=df_5.loc[df_5['level_0'].duplicated(keep=False)]
#df_7=df_5.append(df_6)
#df_7=df_7.drop_duplicates(keep=False)
#df_5=df_5.loc[df_5['level_0'].duplicated(keep=False)]
#
#
#for duplicate in df_6['level_0'].drop_duplicates().tolist():
#    df_7=df_7.append(df_5.loc[df_5.loc[df_5['level_0']==duplicate]['delta_x_lower'].idxmin()])
#
#print(df_7)


#
#print(df_5.loc[df_5['level_0']==10]['delta_x_lower'].min())
#
#df_7=df_7.append(df_5.loc[df_5.loc[df_5['level_0']==10]['delta_x_lower'].idxmin()])
#df_7=df_7.append(df_5.loc[df_5.loc[df_5['level_0']==7]['delta_x_lower'].idxmin()])
#df_7=df_7.append(df_5.loc[df_5.loc[df_5['level_0']==4]['delta_x_lower'].idxmin()])
#df_7=df_7.append(df_5.loc[df_5.loc[df_5['level_0']==1]['delta_x_lower'].idxmin()])
#print(df_7)
#
#
#print(df_5.loc[df_5['level_0']==10]['delta_x_lower'].idxmin())
##df_8=df_5.drop(df_5.loc[df_5['area'] == df_5.loc[df_5['level_0']==10]['delta_x_lower'].idxmin()])\
#print(df_5.loc[df_5['level_0']==10]['delta_x_lower'].idxmax())
#df_8=df_5.loc[df_5.loc[df_5['level_0']==10]['delta_x_lower'].idxmin()]
#print(df_8)
#df_9=df_8.append(df_5.loc[df_5.loc[df_5['level_0']==7]['delta_x_lower'].idxmin()])
#print(df_9)
#
#
#
#
#
#
#
#
#
#
#
#
#print(df_5.loc[0],df_5.loc[1],df_5.loc[2],df_5.loc[3],df_5.loc[4],df_5.loc[5],df_5.loc[6],df_5.loc[7],df_5.loc[8],df_5.loc[9],df_5.loc[10])
##df_5=df_5.loc[df_5['sum']>0]
##df_5=df_5.loc[df_5['delta_x_lower']<650]
##df_5=df_5.loc[df_5['delta_y_lower']+df_5['delta_y_upper']<height*0.7]
#print(df_5.index)
#
#
#df_5=df_5.loc[df_5['delta_y_lower']<250]
#df_5=df_5.loc[df_5['delta_y_upper']>-250]
#
#print(df_5)
#df_6=df_5.loc[df_5.index.duplicated(keep=False)]
#print(df_5.loc[df_5.index.duplicated(keep=False)])
#print(df_6['delta_x_lower'].max())
#df_5=df_5[df_5['delta_x_lower'] != df_6['delta_x_lower'].max()]
#print(df_5)
#
#df_5=calc_perc_off_mean(df_5.copy(),'area','perc_off_mean_area')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_lower','perc_off_mean_yl')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_upper','perc_off_mean_yu')
#df_5=calc_perc_off_mean(df_5.copy(),'ratio','perc_off_mean_ratio')
#df_5=calc_perc_off_mean(df_5.copy(),'sum','perc_off_mean_sum')
#
#
#df_5['perc_sum']=df_5['perc_off_mean_area'].add(df_5['perc_off_mean_yl'].add(df_5['perc_off_mean_yu'].add(df_5['perc_off_mean_ratio'].add(df_5['perc_off_mean_sum'])))).divide([5]*len(df_5.index.tolist()))
#
#
#print(df_5)
#df_5=df_5.drop(df_5['perc_sum'].idxmax())
#print(df_5)
#
#df_5=calc_perc_off_mean(df_5.copy(),'area','perc_off_mean_area')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_lower','perc_off_mean_yl')
#df_5=calc_perc_off_mean(df_5.copy(),'delta_y_upper','perc_off_mean_yu')
#df_5=calc_perc_off_mean(df_5.copy(),'ratio','perc_off_mean_ratio')
#df_5=calc_perc_off_mean(df_5.copy(),'sum','perc_off_mean_sum')
#
#df_5['perc_sum']=df_5['perc_off_mean_area'].add(df_5['perc_off_mean_yl'].add(df_5['perc_off_mean_yu'].add(df_5['perc_off_mean_ratio'].add(df_5['perc_off_mean_sum'])))).divide([5]*len(df_5.index.tolist()))
#print(df_5)
#
#
#
#
#
#
#
#
#
#
#
#score_data,img_2=find_stave_data('source/scores/img_8.png',init_filter_thresh=140,width_thresh=0.2,delta_y_thresh=0.2)
#
#score_data['staves']=score_data['staves'].reset_index(drop=True)
#
#score_data['staves']['delta_upper_bounds']=score_data['staves'].upper_bounds.diff().shift(-1).fillna(0)
#score_data['staves']['delta_lower_bounds']=score_data['staves'].lower_bounds.diff().shift(-1).fillna(0)






#
#cv2.imshow('img',img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
