#!/usr/bin/env python      
# -*- coding: utf-8 -*-
#author:Blue_Bubble
#datetime:2019/1/26 17:10
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os

load_model("model.h5")
print("Loaded model from disk")
model = load_model('model.h5')


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 15

filenames = os.listdir("/home/apps/work/dataset/cat_vs_dog/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


test_filenames = os.listdir("/home/apps/work/dataset/cat_vs_dog/test" )
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    test_filenames = os.listdir("/home/apps/work/dataset/cat_vs_dog/test/" ),
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

threshold = 0.5
test_df['probability'] = predict
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)
# test_df['category'].value_counts().plot.bar()

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    probability = row['probability']
    img = load_img("/home/apps/work/dataset/cat_vs_dog/test/" +filename, target_size=IMAGE_SIZE)
    logging.INFO(filename + 'class-(' + "{}".format(category) + ')' '(' + "{}".format(round(probability, 2)) + ')')





