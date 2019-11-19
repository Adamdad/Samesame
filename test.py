import tensorflow as tf
import numpy as np
e_model_path = "model.h5"
model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
image_name = "elephant.jpg"
img = tf.keras.preprocessing.image.load_img(image_name, target_size=(299, 299))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = tf.keras.applications.inception_v3.preprocess_input(x)


for layer in model.layers:
    if layer.name.startswith("activation"):
        print(layer.get_weights())
