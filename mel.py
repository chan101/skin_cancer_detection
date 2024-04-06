#!pip install -U efficientnet
# import sys
# import os
# from os.path import join
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import efficientnet.keras as efn 
# import tensorflow as tf
#import tensorflow_addons as tf
# from keras.preprocessing.image import load_img, img_to_array
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from efficientnet.tfkeras import EfficientNetB4
import keras
import math
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.layers import Conv2D,BatchNormalization,MaxPool2D,Flatten,Dense
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
EP=90
LR=0.0001
DR=LR/EP
MM=0.8
K=0.2
def lr_scheduler(epoch, lr):return lr* float((np.exp(- (epoch*DR))))
LRD=LearningRateScheduler(lr_scheduler)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=10000,decay_rate=DR)
OPT=keras.optimizers.Adam(learning_rate=LR)



def start_prog():
    model = keras.Sequential()
    model.add(EfficientNetB4(input_shape=(224, 224, 3), include_top=False, weights='imagenet'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu',kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2, activation='sigmoid')) 
    model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy', 'AUC'])
    test_dir = 'static/images/'
    target_size=(224, 224)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,target_size=target_size,batch_size=1)
    print(test_generator.class_indices)
    len(test_generator.labels)
    model.load_weights('best_model.hdf5')
    model.save_weights('model.hdf5')
    model.load_weights('best_model.hdf5')
    y=np.concatenate([test_generator.next()[1] for i in range(test_generator.__len__())])
    true_labels=np.argmax(y, axis=-1)
    prediction= model.predict(test_generator, verbose=2)
    print(prediction)
    percent = prediction[0]*100
    prediction=np.argmax(prediction, axis=-1)


    
    cm = confusion_matrix(y_true=true_labels, y_pred=prediction)
    print(cm)
    if cm[0][0] > 0:
        print("MELANOMA")
        res = "MELANOMA"
    else:
        print("NOT MELANOMA")
        res = "NOT MELANOMA"
 

 
   
    acc=accuracy_score(true_labels,prediction) 
    print('Accuracy: %.3f' % acc)
    precision = precision_score(true_labels,prediction,labels=[1,2], average='micro')
    print('Precision: %.3f' % precision)
    recall = recall_score(true_labels,prediction, average='binary')
    print('Recall: %.3f' % recall)
    score = f1_score(true_labels,prediction, average='binary')
    print('F-Measure: %.3f' % score)
    return [res,percent]
######################################################


import os
import shutil
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
UPLOAD_FOLDER = 'static/images/melanoma'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def move(res):
    if res == "melanoma":
        t_dir = 'new_dataset/melanoma/'
    else:
        t_dir = 'new_dataset/not_melanoma/'
    s_dir = 'static/images/melanoma'
    
    file_names = os.listdir(s_dir)
    
    for file_name in file_names:
        shutil.copy(os.path.join(s_dir, file_name), t_dir)	
@app.route('/')
def upload_form():
    data = {}
    return render_template('index.html',data=data)

@app.route('/', methods=['POST'])
def upload_image():
    shutil.rmtree('static/images/melanoma')
    newpath = 'static/images/melanoma'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    res = start_prog()
    move(str(res[0]))
    
    if str(res[0]) == "MELANOMA":
        perc = math.floor(res[1][0])
        data = {'Melanoma skin cancer' : 'Accuracy', 'Melanoma' : int(perc), 'Not Melanoma' : int(100-perc)}
        return render_template('index.html', data=data, filenames=file_names,result="Prediction is " + str(res[0]) + " and the accuracy is " + str(math.floor(res[1][0])) +"%")
    else:
        perc = math.floor(res[1][1])
        data = {'Melanoma skin cancer' : 'Accuracy', 'Melanoma' : int(100-perc), 'Not Melanoma' : int(perc)}
        return render_template('index.html', data=data, filenames=file_names,result="Prediction is " + str(res[0]) + " and the accuracy is " + str(math.floor(res[1][1])) +"%")
    
    
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='images/melanoma/' + filename), code=301)
    

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

    
######################################################

