import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("vgg16_transfer_1.h5")
#converter.optimizations = [tf.lite.Optimize.OPTION1]
#converter.target_spec.supported_types = [tf.OPTION2]
tflite_model = converter.convert()

with open("vgg16_transfer_1.tflite", "wb") as f:
    f.write(tflite_model)