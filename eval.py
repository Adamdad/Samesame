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
    for batch in tqdm(val_generator.generate()):
        img,label = batch
        preds = model.predict(img)
        top_5 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds,k=5))
        top_1 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds, k=1))
    print("top 1 {}, top 5 {}".format(top_1/num_data,top_5/num_data))

def qevaluate(model,mode="full",model_name = "original"):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if mode=="full":
        num_data, label_to_name, representative_gen = prepare_data(num_data=100,
                                                                   batch_size=1,
                                                                   data_dir="../data/ILSVRC2012_img_val/",
                                                                   val_file="val.txt",
                                                                   mapping_file="synset_words.txt")
        def representative_dataset_gen():
            for batch in tqdm(representative_gen.generate()):
                # Get sample input data as a numpy array in a method of your choosing.
                img,label = batch
                img = np.array(img, dtype=np.float32)
                yield [img]

        converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
    elif mode=="weights":
        pass

    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    open(
        'weights/{}_quantized_model.tflite'.format(model_name),
        'wb').write(tflite_model)

    # load the quantized tf.lite model and test

    interpreter = tf.lite.Interpreter(
        model_path='weights/{}_quantized_model.tflite'.format(model_name))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_data, label_to_name, val_generator = prepare_data(num_data = 'all',
                                                          batch_size=1,
                                                         data_dir = "../data/ILSVRC2012_img_val/",
                                                         val_file ="val.txt",
                                                         mapping_file="synset_words.txt")
    top_5 = 0
    top_1 = 0
    for batch in tqdm(val_generator.generate()):
        img,label = batch
        img = np.array(img, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        top_5 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds,k=5))
        top_1 += np.sum(keras.metrics.top_k_categorical_accuracy(label, preds, k=1))
    print("top 1 {}, top 5 {}".format(top_1/num_data,top_5/num_data))


    '''
    # check the tensor data type
    tensor_details = interpreter.get_tensor_details()
    for i in tensor_details:
        print(i['dtype'], i['name'], i['index'])
    '''


if __name__=="__main__":
    e_model_path= "weights/Inceptionv3_equalized.h5"
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
    model.load_weights(e_model_path)
    # evaluate(model)
    qevaluate(model,mode="weights",model_name = "Inceptionv3_equalized")
