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
    
root_folder="C:\\Users\\siddh\\Desktop\\Casia-B\\CASIA-B-processed"
samples=np.random.choice(range(0,75),size=1,replace=False)
samples=["{0:03}".format(i) for i in samples]
X_train,Y_train,labels= Load_Data(root_folder,samples)

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


u6 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c2)
u6 = tf.keras.layers.concatenate([u6, c1])
c6 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

# Output layer (adjust the number of filters according to your task)
output = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='relu')(c6)

# Apply the DepthMean layer to compute the mean along the depth dimension
averaged_output = DepthMean()(output)


checkpointer = tf.keras.callbacks.ModelCheckpoint('c://Users//siddh//Downloads//model_for_gait.keras', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

model1 = tf.keras.Model(inputs=[inputs], outputs=[averaged_output])
model1.compile(optimizer='adam', loss='mse', metrics=['accuracy','precision'])
model1.fit(X_train, Y_train, epochs=2, batch_size=5, validation_split=0.2,callbacks=[callbacks, checkpointer])
    
model1.summary()


epochs=30
for time in range(epochs):
    print("epoch",time)
    

    
    X_train,Y_train,labels= Load_Data(root_folder,samples)
    checkpointer = tf.keras.callbacks.ModelCheckpoint('c://Users//siddh//Downloads//model_for_gait.keras', verbose=1, save_best_only=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

    
    
    loaded_model = tf.keras.models.load_model('c://Users//siddh//Downloads//model_for_gait.keras')
    
    loaded_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    ### The code fit command is going to fit the model on the new data for every epoch and
    ##the callback will keep updating the model weights 

    loaded_model.fit(X_train, Y_train, epochs=2, batch_size=5, validation_split=0.2,callbacks=[callbacks, checkpointer])
    loaded_model.summary()

    
    preds_train = loaded_model.predict(X_train, verbose=1)
    ix = random.randint(0, len(preds_train)-1)
 
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title("Ground_truth")
    ax2.set_title("Segmented_mask")
    ax2.imshow(np.squeeze(preds_train[ix]))
    ax1.imshow(np.squeeze(Y_train[ix]))

    plt.show()
