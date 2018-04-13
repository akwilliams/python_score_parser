import cv2
import os
import random
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
    
for index in range(46):
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
    
#    cv2.imshow('blackhat',blackhat)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    blackhat_closed=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    blackhat_closed_blr=cv2.GaussianBlur(blackhat_closed,(1,15),4)
        
#    cv2.imshow('test',blackhat_closed)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    show_boxes=blackhat_closed.copy()
    hold_0,contours_0,hierarchy=cv2.findContours(blackhat_closed_blr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    boxes={'x0':[],'y0':[],'x1':[],'y1':[],'area':[],'angle':[]}#,'pixel_mean':[],'pixel_mean_q0':[],'pixel_mean_q1':[],'pixel_mean_q2':[],'pixel_mean_q3':[]}
    for c in contours_0:
        x,y,w,h=cv2.boundingRect(c)
        show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
        boxes['x0'].append(x)
        boxes['y0'].append(y)
        boxes['x1'].append(x+w)
        boxes['y1'].append(y+h)
        boxes['area'].append(h*w)
        if len(c)>4:
            (x,y),(MA,ma),angle=cv2.fitEllipse(c)
        else:
            angle=-1
        boxes['angle'].append(angle)
#        value= bh_closed[5][int(y):int(y+h),int(x):int(x+w)]
#        boxes['pixel_mean'].append(np.mean(value))
#        if w >= 2 and h >= 2:
#            y_0,y_1,x_0,x_1=int(y),int(y+(h/2)),int(x),int(x+(w/2))
#            value=bh_closed[5][y_0:y_1,x_0:x_1]
#            boxes['pixel_mean_q0'].append(np.mean(value))
#            y_0,y_1,x_0,x_1=int(y+(h/2)),int(y+h),int(x),int(x+(w/2))
#            value=bh_closed[5][y_0:y_1,x_0:x_1]
#            boxes['pixel_mean_q1'].append(np.mean(value))
#            y_0,y_1,x_0,x_1=int(y),int(y+(h/2)),int(x+(w/2)),int(x+w)
#            value=bh_closed[5][y_0:y_1,x_0:x_1]
#            boxes['pixel_mean_q2'].append(np.mean(value))
#            y_0,y_1,x_0,x_1=int(y+(h/2)),int(y+h),int(x+(w/2)),int(x+w)
#            value=bh_closed[5][y_0:y_1,x_0:x_1]
#            boxes['pixel_mean_q3'].append(np.mean(value))
#        else:
#            boxes['pixel_mean_q0'].append(-1)
#            boxes['pixel_mean_q1'].append(-1)
#            boxes['pixel_mean_q2'].append(-1)
#            boxes['pixel_mean_q3'].append(-1)
        df=pd.DataFrame(data=boxes)   



    cv2.imshow('show_boxes',show_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    df['count']=df.index.tolist()
    df=df.sort_values(by=['x0','y0','area'],ascending=[True,True,False])
    df=df.reset_index(drop=True)
    
    lib={'x0':[],'y0':[],'x1':[],'y1':[],'area':[],'angle':[]}
    
    for box in range(len(df.index.tolist())-1):
        info=df.iloc[box:box+1,:]
        if ((info['y1'].tolist()[0]-info['y0'].tolist()[0])*(info['x1'].tolist()[0]-info['x0'].tolist()[0])) > 7000 and (info['y1'].tolist()[0]-info['y0'].tolist()[0])/(info['x1'].tolist()[0]-info['x0'].tolist()[0])<4.5 and (info['y1'].tolist()[0]-info['y0'].tolist()[0])/(info['x1'].tolist()[0]-info['x0'].tolist()[0])>1.5 and info['x0'].tolist()[0]<1000:
            lib['x0'].append(info['x0'].tolist()[0])
            lib['y0'].append(info['y0'].tolist()[0])
            lib['x1'].append(info['x1'].tolist()[0])
            lib['y1'].append(info['y1'].tolist()[0])
            lib['area'].append(info['area'].tolist()[0])
            lib['angle'].append(info['angle'].tolist()[0])
            
            print(info['count'].tolist()[0])
            show=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
            cv2.imshow('show',show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    df_2=pd.DataFrame(data=lib)
    print(df_2)
    
    
    
    
    info=df_2.iloc[1:2,:]
    show=img[info['y0'].tolist()[0]:info['y1'].tolist()[0],info['x0'].tolist()[0]:info['x1'].tolist()[0]]
    cv2.imshow('show',show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    w, h = show.shape[::-1]
    methods = ['cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        img2 = img.copy()
        method = eval(meth)
        
        # Apply template Matching
        res = cv2.matchTemplate(img2,show,method)
        #    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        cv2.imshow('res',res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#        top_left = min_loc
#    else:
#        top_left = max_loc
#    bottom_right = (top_left[0] + w, top_left[1] + h)
#
#    cv2.rectangle(img,top_left, bottom_right, 255, 2)
#
#    plt.subplot(121),plt.imshow(res,cmap = 'gray')
#    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(img,cmap = 'gray')
#    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#    plt.suptitle(meth)
#
#    plt.show()



#cv2.imshow('bh_closed',bh_closed[1])
##cv2.imshow('bh_closed',bh_closed[1])
#cv2.imshow('2',bh_closed[10])
##cv2.drawContours(mask,contours_0[824],0,255,-1)
##x,y,w,h=cv2.boundingRect(contours_0[824])
##value= bh_closed[5][y:(y+h),x:(x+w)]
##print(np.mean(value))
##
##cv2.imshow('show_boxes',mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
## Non-flaged, non-bared notes for val in [788,227,228,789,780,425,420,421,423,382,384,348,782,781,783,784,655,645,646,641,610,611,576,575,422,381,383,349,180,178,179,146,147,796,689,690,688,619,584,585,586,587,577,463,363,254,183,184,185,172,155,785,786,613,479,462,442,443,424,399,394,361,362,350,797,798,675,634,593,594,458,459,364,365,366,255,229,230,691,656,657,633,588,589,590,591,592,793,650,651,624,622,626,471,468,466,469,407,357,358,359,240,164,165]:
## Fermatas for val in [809,376,702,670,638,638,265,218,177,833,808,701,669,637,606,605,489,490,456,457,419,418,377,264,217,175]:
#
#parse = df.iloc[[788,227,228,789,780,425,420,421,423,382,384,348,782,781,783,784,655,645,646,641,610,611,576,575,422,381,383,349,180,178,179,146,147,796,689,690,688,619,584,585,586,587,577,463,363,254,183,184,185,172,155,785,786,613,479,462,442,443,424,399,394,361,362,350,797,798,675,634,593,594,458,459,364,365,366,255,229,230,691,656,657,633,588,589,590,591,592,793,650,651,624,622,626,471,468,466,469,407,357,358,359,240,164,165],:]
#
#show_boxes=bh_closed[5].copy()
#for val in [788,227,228,789,780,425,420,421,423,382,384,348,782,781,783,784,655,645,646,641,610,611,576,575,422,381,383,349,180,178,179,146,147,796,689,690,688,619,584,585,586,587,577,463,363,254,183,184,185,172,155,785,786,613,479,462,442,443,424,399,394,361,362,350,797,798,675,634,593,594,458,459,364,365,366,255,229,230,691,656,657,633,588,589,590,591,592,793,650,651,624,622,626,471,468,466,469,407,357,358,359,240,164,165]:
#    x,y,w,h=cv2.boundingRect(contours_0[val])
#    show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
#
##cv2.imshow('show_boxes',show_boxes)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#cv2.imwrite('test.jpg',show_boxes)
#blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
#blackhat_close=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
#blackhat_close_1=cv2.morphologyEx(blackhat_close,cv2.MORPH_CLOSE,np.ones((4,1),np.uint8))
#blackhat_close_2=cv2.morphologyEx(blackhat_close_1,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#blackhat_close_3=cv2.morphologyEx(blackhat_close_2,cv2.MORPH_CLOSE,np.ones((6,1),np.uint8))
#blackhat_open_1=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_OPEN,np.ones((2,4),np.uint8))
#blackhat_close_4=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
#
#blackhat_close_5=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#difference=cv2.subtract(blackhat_close_5,blackhat_close_4)
#show_boxes=blackhat_close_4.copy()
#
#hold_0,contours_1,hierarchy=cv2.findContours(blackhat_close_4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours_1))
#for c in contours_1:
#    x,y,w,h=cv2.boundingRect(c)
#    show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
#        
#    #
##cv2.imshow('img_threshold',img_threshold)
##cv2.imshow('img_close',img_close)
##cv2.imshow('blackhat',blackhat)
##cv2.imshow('blackhat_close',blackhat_close)
##cv2.imshow('blackhat_close_1',blackhat_close_1)
##cv2.imshow('blackhat_close_2',blackhat_close_2)
##cv2.imshow('blackhat_close_3',blackhat_close_3)
##cv2.imshow('blackhat_close_4',blackhat_close_4)
##cv2.imshow('blackhat_close_5',blackhat_close_5)
##cv2.imshow('show_boxes',show_boxes)
#        #
##cv2.imshow('difference',difference)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
    
    
    
    
    
#    
#    
#    for index in range(15):
#    string='source/scores/img_'+str(index+25)+'.png'
#    print(string,index)
#    img=cv2.imread(string,cv2.IMREAD_GRAYSCALE)
#    height,width=img.shape[:2]
#    filter_arg=int(width*0.5)
#    filter_arg=filter_arg if filter_arg%2==1 else filter_arg+1
#    img_filtered=cv2.GaussianBlur(img,(filter_arg,1),0)
#    th, img_threshold=cv2.threshold(img_filtered,190,255,cv2.THRESH_BINARY_INV)
#    
#    img_open=cv2.morphologyEx(img_threshold,cv2.MORPH_OPEN,np.ones((1,1111),np.uint8))
#    img_close=cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    
#    
#    th,img_threshold_1=cv2.threshold(img,190,255,cv2.THRESH_BINARY_INV)
#    hold_0,contours_0,hierarchy=cv2.findContours(img_threshold_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    print(len(contours_0))
#    blackhat=cv2.subtract(img_threshold_1,img_close)
#    
#    
#    blackhat_open=cv2.morphologyEx(blackhat,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
#    blackhat_close=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
#    blackhat_close_1=cv2.morphologyEx(blackhat_close,cv2.MORPH_CLOSE,np.ones((4,1),np.uint8))
#    blackhat_close_2=cv2.morphologyEx(blackhat_close_1,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    blackhat_close_3=cv2.morphologyEx(blackhat_close_2,cv2.MORPH_CLOSE,np.ones((6,1),np.uint8))
#    blackhat_open_1=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_OPEN,np.ones((2,4),np.uint8))
#    blackhat_close_4=cv2.morphologyEx(blackhat_close_3,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
#    
#    blackhat_close_5=cv2.morphologyEx(blackhat_open,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
#    difference=cv2.subtract(blackhat_close_5,blackhat_close_4)
#    show_boxes=blackhat_close_4.copy()
#    
#    hold_0,contours_1,hierarchy=cv2.findContours(blackhat_close_4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    print(len(contours_1))
#    for c in contours_1:
#        x,y,w,h=cv2.boundingRect(c)
#        show_boxes=cv2.rectangle(show_boxes,(x,y),(x+w,y+h),(155,155,155),2)
#        
#        #
#    #cv2.imshow('img_threshold',img_threshold)
#    #cv2.imshow('img_close',img_close)
#    #cv2.imshow('blackhat',blackhat)
#    #cv2.imshow('blackhat_close',blackhat_close)
#    #cv2.imshow('blackhat_close_1',blackhat_close_1)
#    #cv2.imshow('blackhat_close_2',blackhat_close_2)
#    #cv2.imshow('blackhat_close_3',blackhat_close_3)
#    #cv2.imshow('blackhat_close_4',blackhat_close_4)
#    #cv2.imshow('blackhat_close_5',blackhat_close_5)
#    cv2.imshow('show_boxes',show_boxes)
#        #
#    #cv2.imshow('difference',difference)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    
#        