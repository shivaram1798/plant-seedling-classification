import os
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.models import load_model
import tensorflow as tf
graph = tf.get_default_graph()
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


def loading_model():
   # model=load_model('model.h5')
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('model.h5')
    print('model loaded')
    return model

def predcit(url,model):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    img=np.array(img.resize((128,128)))
    img=img.reshape(1,128,128,3)
    img = np.array(img, dtype="float") / 255.0
    print(img)
    with graph.as_default():
        b=model.predict(img)
    label=np.argmax(b)
    label_name = ['Black -grass', 'Charlock','Cleavers' ,'Common Chickweed','Common Wheat' ,'Fat Hen','Loose Silky-bent' ,'Maize','Scentless Mayweed','Shepherds Purse','Small -floweres cranesbill' ,'Sugar beet']
    #return label
    return label_name[label]
        


