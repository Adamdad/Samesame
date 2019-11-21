import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import plot_model

def visual_weight(model):
    for layer in model.layers:
        # print(layer.name)
        if layer.name.startswith("conv"):
            weight = np.array(layer.get_weights())
            b,h,w,in_channel,out_channel = weight.shape
            fig = plt.figure()
            for i in range(out_channel):
                c = '#%02X%02X%02X' % (random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))
                ax = fig.add_subplot()
                n, bins, patches = ax.hist(x=weight[:,:,:,:,i].reshape(b*h*w*in_channel), bins='auto', color=c,
                                            alpha=0.7, rwidth=0.8)
                ax.grid(axis='y', alpha=0.75)
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Weight visualization for {} layer'.format(layer.name))
            plt.show()

def visual_graph(model,fig_name='model.png'):
    plot_model(model, to_file=fig_name)

def visual_activation(model,x):
    for layer in model.layers:
        if layer.name.startswith("activation"):
            layer_model = keras.Model(inputs=model.input,
                                outputs=model.get_layer(layer.name).output)
            y = layer_model(x)
            b,h,w,c = y.shape
            for i in range(c):
                c = '#%02X%02X%02X' % (random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))
                y = np.array(y)
                n, bins, patches = plt.hist(x=y[:, :, :,i].reshape(b*h*w), bins='auto', color=c,
                                           alpha=0.7, rwidth=0.8)
                plt.grid(axis='y', alpha=0.75)
                plt.xlabel('Activation Value')
                plt.ylabel('Frequency')
                plt.title('Activation visualization for {}'.format(layer.name))
            plt.show()

model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
visual_graph(model)