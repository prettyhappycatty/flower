from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import sys
import io, base64

graph = tf.compat.v1.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224
    MODEL_FILE_PATH = './flower/ml_models/vgg16_transfer.tflite'
   # MODEL_FILE_PATH = './flower/ml_models/vgg16_transfer.h5'


    classes = ["dandelion", "wood sorrel","pansy","trifolium repens","majalis","azalea","viola mandshurica","hyacinth","rhoeas","cherry blossom"]
    classes_jp = ["タンポポ", "カタバミ","パンジー","シロツメクサ","スズラン","ツツジ","スミレ","ヒヤシンス","ヒナゲシ","サクラ"]
    num_classes = len(classes)

    def predict(self):
        model = None
        global graph
        with graph.as_default():
            #model = load_model(self.MODEL_FILE_PATH)

            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))

            temp_img_array=img_to_array(image)
            img=temp_img_array.astype('float32')/255.0
            img=temp_img_array.reshape((1,self.IMAGE_SIZE,self.IMAGE_SIZE,3))

            #model = load_model('./vgg16_transfer.h5')
            interpreter = tf.lite.Interpreter(model_path=self.MODEL_FILE_PATH)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # add N dim
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()


            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)

            predicted = output_data[0].argmax()
            percentage = int(output_data[0][predicted] * 100)
            #print(classes[predicted], percentage)
            return self.classes_jp[predicted], percentage



    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            return 'data:'+img.file.content_type+';base64,'+base64_img