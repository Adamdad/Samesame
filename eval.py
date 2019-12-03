import tensorflow as tf
import os
import numpy as np
import tensorflow.keras as keras
from utils import prepare_data
from tqdm import tqdm


def evaluate(model):
    num_data, label_to_name, val_generator = prepare_data(num_data = "all",
                                                          batch_size=50,
                                                         data_dir = "../data/ILSVRC2012_img_val/",
                                                         val_file ="val.txt",
                                                         mapping_file="synset_words.txt")
    top_5 = 0
    top_1 = 0
    for batch in tqdm(val_generator):
        img,label = batch
        preds = model.predict(img)
        top_5 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds,k=5))
        top_1 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds, k=1))
    print("top 1 {}, top 5 {}".format(top_1/num_data,top_5/num_data))


if __name__=="__main__":
    e_model_path= "model.h5"
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
    model.load_weights(e_model_path)
    evaluate(model)
