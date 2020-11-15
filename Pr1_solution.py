import numpy as np
import pandas as pd
import tensorflow as tf

import os
import cv2

import glob
import itertools
from sklearn.model_selection import train_test_split

from pathlib import Path

import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Input, Dense, Flatten
#from keras.models import Model, load_model

import shutil
import datetime
import time
import requests
import glob

def create_dataset(img_folder, IMG_HEIGHT, IMG_WIDTH):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        print (os.path.join(img_folder, dir1))
        i = 0
        for file in os.listdir(os.path.join(img_folder, dir1)):
            if file.endswith("jpg") and i < 6:
              print (file)
              i+=1

              image_path= os.path.join(img_folder, dir1,  file)
              image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
              image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
              image=np.array(image)
              image = image.astype('float32')
              image /= 255 
              img_data_array.append(image)
              class_name.append(dir1)
    return img_data_array, class_name


def create_inference_dataset(img_folder, IMG_HEIGHT, IMG_WIDTH):
        filename_array = []
        img_data_array=[]
        i = 0
        for file in os.listdir(img_folder):
            if file.endswith("jpg") and i < 6:
              print (file)
              i+=1
              filename_array.append(file)
              image_path= os.path.join(img_folder,  file)
              image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
              image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
              image=np.array(image)
              image = image.astype('float32')
              image /= 255 
              img_data_array.append(image)
        return img_data_array, filename_array

#%% Define model
def cifar10_model(input_shape, n_classes):

    #input_tensor = Input(shape=input_shape)
    mdl =tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

    #mdl = Model(inputs=base_model.input, outputs=predictions)
    mdl.build(input_shape=(None, 240, 320, 3))
    mdl.summary()
    return mdl
    
#%%
def load_checkpoint_model(checkpoint_path, checkpoint_names):
    list_of_checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*'))
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in list_of_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         checkpoint_names.format(epoch=checkpoint_epoch_number))
    resume_model = load_model(checkpoint_epoch_path)
    return resume_model, checkpoint_epoch_number
    
#%%
def define_callbacks(volume_mount_dir, checkpoint_path, checkpoint_names, today_date):

    # Model checkpoint callback
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    filepath = os.path.join(checkpoint_path, checkpoint_names)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')

    # Loss history callback
    epoch_results_callback = CSVLogger(os.path.join(volume_mount_dir, 'training_log_{}.csv'.format(today_date)),
                                       append=True)

    class SpotTermination(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs={}):
            status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
            if status_code != 404:
                time.sleep(150)
    spot_termination_callback = SpotTermination()

    callbacks = [checkpoint_callback, epoch_results_callback, spot_termination_callback]
    return callbacks    
    
#%%
def main():

    # Training parameters
    batch_size = 512
    epochs = 50
    volume_mount_dir = '/dltraining/'
    dataset_path = os.path.join(volume_mount_dir, 'datasets')
    checkpoint_path = os.path.join(volume_mount_dir, 'checkpoints')
    checkpoint_names = 'cifar10_model.{epoch:03d}.h5'
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    awskey = 'AKIASDWXX2IZTRXQUT6P'
    awssecret = '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'
    conn = S3Connection(awskey, awssecret)

    train_dir = conn.lookup('train')
    test_dir = conn.lookup('test')

    IMG_WIDTH=240
    IMG_HEIGHT=320

    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    
    # Load dataset
    # extract the image array and class name
    img_data, class_name = create_dataset(train_dir, IMG_HEIGHT, IMG_WIDTH)        
    inference_data, inference_keys = create_inference_dataset(test_dir, IMG_HEIGHT, IMG_WIDTH)
    print (len(inference_data))
    #class_name

    target_dict={k: v for v, k in enumerate(np.unique(class_name))}
    target_dict

    img_data = np.array(img_data, np.float32)
    target_val= np.array([target_dict[class_name[i]] for i in range(len(class_name))])
    
    inference_data = np.array(inference_data, np.float32)
    
    print (inference_data.shape)
    print (img_data.shape)
    print (target_val.shape)

    n_classes = np.unique(class_name).shape[0]

    train_x,test_x, train_y, test_y = train_test_split(img_data,target_val, test_size=0.2)

    print (len(train_x))
    print (len(train_y))
    print (len(test_x))
    print (len(test_y))

    # Load model
    if os.path.isdir(checkpoint_path) and any(glob.glob(os.path.join(checkpoint_path, '*'))):
        model, epoch_number = load_checkpoint_model(checkpoint_path, checkpoint_names)
    else:
        model = cifar10_model(input_shape, n_classes)
        epoch_number = 0

    # Define Callbacks
    callbacks = define_callbacks(volume_mount_dir, checkpoint_path, checkpoint_names, today_date)

    model.compile(optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, initial_epoch=epoch_number, callbacks=callbacks)

    # Score trained model.
    scores = mmodel.evaluate(test_x, test_y)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Backup terminal output once training is complete
    shutil.copy2('/var/log/cloud-init-output.log', os.path.join(volume_mount_dir,
                                                                'cloud-init-output-{}.log'.format(today_date)))
                                                                
    prediction_inference = model(inference_data)
    inference_answer = [list(target_dict.keys())[list(target_dict.values()).index(np.argmax(i))] for i in prediction_inference]
    res_df = pd.DataFrame(zip(inference_keys,inference_answer), columns = ['Filename','Class'])
    res_df.to_csv('ekaterina.bogorada_project01.csv',index = False)
    
    
if __name__ == "__main__":
    main()    
   
