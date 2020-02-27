# extract and plot each detected face in a photograph
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import os


file=[]
files=[]
path = 'path/to/faces/'
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))
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
    
      if x1>0 and y1 >0:
            plt.subplot(1, len(result_list), i+1)
            plt.axis('off')
            plt.imshow(data[y1-30:y2+30,x1-30:x2+30])
            plt.savefig(save,dpi=200)

##Remove jpg in face file
for r, d, f in os.walk(path):
    for folder in d:
        folders.append(os.path.join(r, folder))
for f in folders:
   for file in os.listdir(f):
   
    if file.endswith('.jpg'):
       os.remove(f+'/'+file)



import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import hog
from skimage import data, exposure
import numpy as np
import glob

dff=pd.DataFrame()




folders = []
#reading folders path in face folder
for r, d, f in os.walk(path):
    for folder in d:
        folders.append(os.path.join(r, folder))



image=[]
names=[]
name=[]
images=[]


for f in folders:
    #reading png images in each folder
    images = [cv2.imread(file) for file in glob.glob(f+'/*.png')]
      
    for i in range(len(images)):
    #Reading folder name for labeling
     name=os.path.basename(f)
     names=np.append(names,name)
     #resizing image for HOG
     image=cv2.resize(images[i],(128,64))
     #Implementing HOG
     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
                    

     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

     ax1.axis('off')
     ax1.imshow(image, cmap=plt.cm.gray)
     ax1.set_title('Input image')

     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

     data=hog_image_rescaled.reshape(1,-1)
     data=pd.DataFrame(data)
     dff=dff.append(data)

     ax2.axis('off')
     ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
     ax2.set_title('Histogram of Oriented Gradients')
     plt.show()




dff1=pd.DataFrame(names)


x_train=dff.iloc[:,:].values
y_train=dff1.iloc[:,0].values
x_test=dff.iloc[22:35,:].values
y_test=dff1.iloc[22:35,:].values


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#Save model
import pickle
pickle.dump(classifier,open('MS.sav','wb'))
pickle.dump(sc,open('scaler.pkl','wb'))



