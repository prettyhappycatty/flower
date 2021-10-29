import numpy as np
#from tensorflow import keras
#from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow as tf
from PIL import Image
import sys
from keras.preprocessing.image import img_to_array, load_img

classes = ["dandelion", "wood sorrel","pansy","trifolium repens","majalis","azalea","viola mandshurica","hyacinth","rhoeas","cherry blossom"]

num_classes = len(classes)
image_size = 224

#画像を配列に変換し0-1で正規化
image = Image.open(sys.argv[1])
image = image.convert("RGB")
image = image.resize((image_size, image_size))

temp_img_array=img_to_array(image)
img=temp_img_array.astype('float32')/255.0
img=temp_img_array.reshape((1,image_size,image_size,3))

#model = load_model('./vgg16_transfer.h5')
interpreter = tf.lite.Interpreter(model_path='./vgg16_transfer_1.tflite')
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
print(classes[predicted], percentage)