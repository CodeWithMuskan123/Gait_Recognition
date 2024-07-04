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
X_train, Y_train, labels =Load_Data(root_folder, samples)

#####model architecture: for embeddings
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

##change the output according to the requirement , average the layers to produce a single output 
p2_flattened = tf.keras.layers.Flatten()(p2)
fc1 = tf.keras.layers.Dense(256, activation='relu')(p2_flattened)
embedding_model = tf.keras.Model(inputs=inputs, outputs=fc1)

optimizer = tf.keras.optimizers.Adam()

num_epochs = 3  # Number of epochs to train
num_iterations = 3  # Number of times to repeat training with different datasets


for iteration in range(num_iterations):
    
    X_train, Y_train, labels = Load_Data(root_folder,samples)
    
    # Load previous weights
    if iteration > 0:
        embedding_model.load_weights(f'embedding_model_iteration_{iteration}.weights.h5')
    
    
    for epoch in range(num_epochs):
        # Training step
        with tf.GradientTape() as tape:
            embeddings = embedding_model(X_train, training=True)
            loss_value, _ = triplet.batch_all_triplet_loss(labels, embeddings, margin=0.5)
        
        gradients = tape.gradient(loss_value, embedding_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))
    
    # Save the model weights after each iteration
    embedding_model.save_weights(f'embedding_model_iteration_{iteration + 1}.weights.h5')