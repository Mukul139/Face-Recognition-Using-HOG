# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:21:40 2020

@author: DELL
"""



from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import os
import glob
import pickle
#load 
scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('MS.sav','rb'))

import pandas as pd
path = 'path/to/test/'


df_test=pd.DataFrame()

folders = []

image=[]
names=[]
name=[]
images=[]
file=[]
files=[]

for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

#detect face
for f in files:
    filename = f
    save=filename[:-3] + 'png'
    pixels =plt.imread(filename)
    detector = MTCNN()
    result_list= detector.detect_faces(pixels)
    
    data = plt.imread(filename)
    for i in range(len(result_list)):
      x1,y1, width, height = result_list[i]['box']
      x2, y2 = x1 + width, y1 + height  
    
      if x1>0 and y1>0:
            plt.subplot(1, len(result_list), i+1)
            plt.axis('off')
            plt.imshow(data[y1-30:y2+30,x1-30:x2+30])
            plt.savefig(save,dpi=200)

 
            images = [cv2.imread(file) for file in glob.glob(path+'/*.png')]
   #Hog and classifier         
            for i in range(len(images)):
                     image=cv2.resize(images[i],(128,64))
                    
                     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
                                    
                    
                     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                    
                     ax1.axis('off')
                     ax1.imshow(image, cmap=plt.cm.gray)
                     ax1.set_title('Input image')
                    
                    # Rescale histogram for better display
                     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                    
                     data=hog_image_rescaled.reshape(1,-1)
                    
                     data=pd.DataFrame(data)
                     df_test=df_test.append(data)
                    
                     ax2.axis('off')
                     ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                     ax2.set_title('Histogram of Oriented Gradients')
                     plt.show()
                    
                     x_test=df_test.iloc[:,:].values
                     x_test=scaler.transform(x_test)
                     
                     y_pred_prob=model.predict_proba(x_test)
                     y_pred=model.predict(x_test)
                     y_pred=pd.DataFrame(y_pred)
                    
                     z=0
                     y=0
                     y_pred_prob=pd.DataFrame(y_pred_prob)
                   
                     for i in range(len(y_pred_prob.columns)):
                        if (y_pred_prob[i].values)>0.5:
                       
                           z=i
                           y=y_pred_prob[i].values
                     if y>0:
                        print(y_pred[0].to_csv(index=False))
                        
                     else:
                        print("Sorry probability didn't reach threshold value")   

      else:
          print('No Face detected')
        

files=[]      
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))   

for f in files:
    if f.endswith('.png'):
       os.remove(f)
              
       





