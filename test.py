import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers, models
import os
import random
import numpy as np
from sklearn.manifold import TSNE
import cv2

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import triplet


IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3


@register_keras_serializable()
class DepthMean(tf.keras.layers.Layer):
    def call(self, inputs):
        # Compute the mean along the depth dimension (axis=1)
        averaged_output = tf.reduce_mean(inputs, axis=1)
        return averaged_output



def Load_Data(path,samples):
    X_train=[]
    Data=[]
    labels=[]
    for sample in samples:
        angles_selected=[]
        z=3
        for _ in range(z):
            frames=[]
            types=['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
            t=np.random.choice(types)
            angles=os.listdir(os.path.join(path,sample,t))
            
            angle=np.random.choice(angles)
            
            while (angle in angles_selected) and (len(angles_selected)<3):
                angle=np.random.choice(angles)
                
                
            
            x_train=[]
            length= len(os.listdir(os.path.join(path,sample,t,angle)))
            if length <30:
                z+=1
                continue  
            for image in os.listdir(os.path.join(path,sample,t,angle)):
                if image is not None:
                    file_path=os.path.join(path,sample,t,angle,image)
                
                    img =cv2.imread(file_path)
                
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
                    x_train.append(img)
                    img=resize(img,(IMG_HEIGHT,IMG_WIDTH,1),mode='constant', preserve_range=True)
                
                
                
                    frames.append(img)
                    
                    if len(x_train)==30:
                    
                        break
                else:
                    break
            angles_selected.append(angle)
            X_train.append(x_train)
            Data.extend([np.average(frames,axis=0)])
            labels.append([sample])
            #print(np.average(frames,axis=0))
        
        
    
    return np.array(X_train), np.array(Data), np.array(labels)
    
def Load_all(path,all_data):
    X_train=[]
    labels=[]
    
    for sample in all_data:
        
        types=['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        for t in types:
            angles=os.listdir(os.path.join(path,sample,t))
            angle='018'
            for angle in angles:
                x_train=[]
                
                length= len(os.listdir(os.path.join(path,sample,t,angle)))
                if length <30:
                
                    continue  
                for image in os.listdir(os.path.join(path,sample,t,angle)):
                    if image is not None:
                        file_path=os.path.join(path,sample,t,angle,image)
                
                        img =cv2.imread(file_path)
                
                        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
                        x_train.append(img)
                        
                    
                        if len(x_train)==30:
                    
                            break            
                X_train.append(x_train)
                labels.append([sample])    
    return np.array(X_train), np.array(labels)

                
        

test_folder="C:\\Users\\siddh\\Desktop\\Casia-B\\CASIA-B-test"

samples=np.random.choice(range(75,85),size=1,replace=False)
samples=["{0:03}".format(i) for i in samples]
all_data= ["{0:03}".format(i) for i in range(75,85)]
print("Random Probe",samples)

#X_train, Y_train, labels =Load_Data(root_folder, samples)
#print("labels",labels)
#np.save('X_train.npy',X_train)
#np.save('Y_train.npy',Y_train)
#np.save('labels.npy', labels)

X_test, Y_test, labels = Load_Data(test_folder,samples)
X_all,labels_all=Load_all(test_folder,all_data)

inputs = tf.keras.Input(shape=(30, 64, 64, 1))
inputs=tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)
 
p2_flattened = tf.keras.layers.Flatten()(p2)
fc1 = tf.keras.layers.Dense(256, activation='relu')(p2_flattened)
embedding_model = tf.keras.Model(inputs=inputs, outputs=fc1)
   
embedding_model.load_weights('c://Users//siddh//Downloads//embedding_model_iteration_30.weights.h5')
       
embeddings_test = embedding_model.predict(X_test)
embeddings_all= embedding_model.predict(X_all)
print("Shape of both the embeddings",embeddings_test.shape,embeddings_all.shape)

distances=[]
for e in embeddings_all:
    distances.append(np.linalg.norm(embeddings_test[0] - e))

index=np.argmin(distances)

print("The probe matches best with person ID:",labels_all[index], "and the distance came out to be:", distances[index])
fig, (ax1,ax2)=plt.subplots(1,2)
ax1.set_title("single frame from given probe")
ax2.set_title("Recognized gallery image")
ax1.imshow(np.squeeze(X_test[0][0]))
ax2.imshow(np.squeeze(X_all[index][0]))
plt.show()



    
loaded_model = tf.keras.models.load_model('c://Users//siddh//Downloads//model_for_gait.keras')
    

preds_test=loaded_model.predict(X_test)
ix = random.randint(0, len(preds_test)-1)

plt.show()
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_title("Ground_truth")
ax2.set_title("Segmented_mask")
ax2.imshow(np.squeeze(preds_test[ix]))
ax1.imshow(np.squeeze(Y_test[ix]))

plt.show()


   

    



