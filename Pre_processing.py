import cv2
import numpy as np
import h5py
import pandas as pd
import os


def convert_to_gs(path,samples):
    
    for sample in samples:
       
        types=['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        
        for t in types:
            angles=os.listdir(os.path.join(path,sample,t))
            for angle in angles:
                i=1
                for image in os.listdir(os.path.join(path,sample,t,angle)):
                    
                    img=cv2.imread(os.path.join(path,sample,t,angle,image))
                    
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    if gray_image is not None:
                        cv2.imwrite(os.path.join(path,sample,t,angle, f'{i:03}.jpg'), gray_image)
                        i+=1
                        
                        

def load_images_from_folder(folder,samples):
    
    
    
    
    for sample in samples:
        lost=[]
        j=1
        types=['bg-01','bg-02','cl-01','cl-02','nm-01','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        
        for t in types:
            angles=os.listdir(os.path.join(folder,sample,t))
            for angle in angles:
                i=1
                for image in os.listdir(os.path.join(folder,sample,t,angle)):
                    img=cv2.imread(os.path.join(folder,sample,t,angle,image))
                    if img is not None:
                        x1, y1, w1, h1 = (0,0,0,0)
                        points = 0

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
                        points = np.argwhere(thresh_gray==0) # find where the black pixels are
                        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
                        x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
                        crop = img[y:y+h, x-30:x+w+30]
                        try:
                            resized=cv2.resize(crop, (64,64), interpolation = cv2.INTER_AREA)
                            cv2.imshow( "crop",crop)
                            cv2.imshow("resized",resized)
                            
                            newpath = f'C:/Users/siddh/Desktop/Casia-B/CASIA-B-processed/{sample}/{t}/{angle}'
                            if not os.path.exists(newpath):
                                os.makedirs(newpath)
                            cv2.imwrite(os.path.join(newpath , f'{i:03}.jpg'), resized)
                            i+=1
                        except cv2.error:
                            #print(os.path.join(folder,sample,t,angle,image))
                            
                            lost.append(j)
                            
                            j+=1
                            pass
        
            print("samples lost",len(lost))
            
                        
    return
                                            



        
            
                
        




samples=[i for i in range(1,75)]
samples=["{0:03}".format(i) for i in samples]


folder="C:/Users/siddh/Desktop/Casia-B/CASIA-B"
path='C:/Users/siddh/Desktop/Casia-B/CASIA-B-processed'


Input=load_images_from_folder(folder,samples)
convert_to_gs(path,samples)


    
    


