import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def load_image(image_name):
    img = keras.preprocessing.image.load_img(image_name, target_size=(299, 299))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.inception_v3.preprocess_input(x)
    return x

def imagenet_generator_epoch(dataset, batch_size=32, num_classes=1000, is_training=False):
    images = np.zeros((batch_size, 299, 299, 3))
    labels = np.zeros((batch_size, num_classes))
    count = 0

    for sample in dataset:
        image = sample["image"]
        label = sample["label"]

        images[count % batch_size] = load_image(image)
        labels[count % batch_size] = np.expand_dims(keras.utils.to_categorical(label, num_classes=num_classes), 0)

        count += 1

        if (count % batch_size == 0):
            yield images, labels

def show_image(img):
    img = img[0]
    img = (img+1)/2
    plt.imshow(img)
    plt.show()

def prepare_data():
    num_data = 100
    data_dir = "../data/ILSVRC2012_img_val/"
    dataset = [{"image": os.path.join(data_dir, line.split()[0]),
                "label": line.split()[1]} for line in open("val.txt").readlines()]
    label_to_name = [line.split()[0] for line in open("synset_words.txt").readlines()]
    random.shuffle(dataset)
    val_generator = imagenet_generator_epoch(dataset[:num_data], batch_size=50, num_classes=1000, is_training=False)
    print("{} images with {} classes".format(num_data, len(label_to_name)))
    return num_data, label_to_name, val_generator