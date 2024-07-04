

# P-01-E Gait recognition : pytorch-3dunet



## Input Data Format
The input data can be downloaded for CASIA-B. It has 125 persons, 75 are used for training and the rest are for testing.
The dataset can be processed using "Preprocessing.py". It will crop and resize all the frames to be 64x64. 


## Train
There are two different models to be trained. The segmentation model and the embedding model.
In order to train on your own data just provide the paths to your training dataset in the respective python files: "segmentation_model.py" and "embedding_model.py". Alternatively you can just load weights for pre-trained model which are saved in "model_for_gait.keras" and "embedding_model_iteration_30.weights.h5."


## Prediction

After saving the checkpoints in a file for latest trained model, run the "test.py" to predict the results for both the models on your test data. Change the path to your latest saved weights before running the code. 
