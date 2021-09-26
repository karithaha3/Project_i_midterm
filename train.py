import  tensorflow as tf 
import  matplotlib.pyplot as plt 
from    cv2 import cv2
import  os
import  numpy as np
from    tensorflow.keras.preprocessing.image import ImageDataGenerator
from    tensorflow.keras.preprocessing import image
import  matplotlib.image as mpimg
from    PIL import ImageOps, Image, ImageDraw, ImageFont



train = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory(
    r'C:\Users\Admin\Desktop\dataset\train',
    target_size =(200,200),
    batch_size = 3,
    class_mode ='categorical'
)

model = tf.keras.models.Sequential([
    #
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu' , input_shape = (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation = 'relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
    metrics = ['accuracy']
)

model.fit(
    train_dataset,
    steps_per_epoch = 3,
    epochs = 30,
    
)

model.save('Phuketfood')

print(train_dataset.class_indices)

