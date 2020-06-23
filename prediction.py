import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
# Import OpenCV
import cv2

# Utility
import itertools
import random
from collections import Counter
from glob import iglob

model=load_model('acc9275own.h5')




with open('categories.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())
    
#print (classes)

IMAGE_SIZE=(224,224)





def load_image(filename):
    img = cv2.imread(filename)
    #img = cv2.imread(os.path.join(image_dir, filename)) #<-- use in case of test through existing validation dataset
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255
    
    return img


def predict(image):
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


#print('function added')

print(model.summary())