from flask import Flask,  render_template, jsonify, request
import datetime
import matplotlib.gridspec as gridspec
from PIL import Image
import os  # working with file system
import tensorflow as tf
import numpy as np
from glob import glob  # for working with files
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import shutil  # for working with files
from matplotlib import image
from matplotlib import pyplot
import time
from tensorflow.keras import layers
from tensorflow.keras import activations


app = Flask(__name__)


# def get_model():
#     model = load_model('generator.h5')
#     return model


generator = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=['POST'])
def generate():

    image_main_collage = []
    for i in range(0, 8):
        image_collage = []
        for j in range(0, 8):
            generated_image = generator(
                tf.random.normal([1, 100]), training=False)
            img = ((generated_image[0, :, :, :] + 1.) / 2.).numpy()
            image_collage.append(img)

        image_main_collage.append(image_collage)

    img_main = concat_tile(image_main_collage)

    print(img)
    t = str(time.time()).split('.')
    filename = 'generated_'+t[0] + '_' + t[1] + '.png'
    path = './static/generated_images/'+filename
    tf.keras.preprocessing.image.save_img(
        path,
        img_main,
        scale=True
    )

    return filename


def concat_tile(im_list_2d):
    return np.vstack([np.hstack(im_list_h) for im_list_h in im_list_2d])


if __name__ == '__main__':
    generator = tf.keras.models.load_model(
        './models/generator.h5', compile=False)
    app.run(debug=True)
