import os
import boto3
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from keras.models import load_model

# AWS S3 config
S3_BUCKET = 'plantseedlingclassification'
S3_KEY = 'model.h5'
LOCAL_MODEL_PATH = 'model.h5'

def download_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_PATH):
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
        print('Downloaded model from S3.')
    else:
        print('Model already exists locally.')

download_model_from_s3()
model = load_model(LOCAL_MODEL_PATH)

app = Flask(__name__, template_folder='templates', static_folder='static')

LABELS = ['Black -grass', 'Charlock','Cleavers' ,'Common Chickweed','Common Wheat' ,'Fat Hen','Loose Silky-bent' ,'Maize','Scentless Mayweed','Shepherd\'s Purse','Small -floweres cranesbill' ,'Sugar beet']

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    link = request.form['in']
    response = requests.get(link)
    img = Image.open(BytesIO(response.content)).resize((128,128))
    img = np.array(img).reshape(1,128,128,3).astype('float32') / 255.0
    pred = model.predict(img)
    label = LABELS[np.argmax(pred)]
    return render_template('index.html', pred=label)

#if __name__ == '__main__':
 #    app.run(debug=True)
