import numpy as np
import tensorflow.keras as keras
import random
from tqdm import tqdm
import os
from utils import load_image,prepare_data
import argparse
from eval import evaluate
keras.backend.set_learning_phase(0)
class Equalizer:
    def __init__(self,model,e_model_path,max_thresh):
        self.model = model
        self.max_thresh = max_thresh
        self.model_path = e_model_path

    def eval(self):
        '''
        Set the model to untrainable
        '''
        for layer in self.model.layers:
            layer.trainable = False

    def get_features(self,x):
        '''
        Get the input feature map for each conv_layer
        :param x: input image
        :return: a dict of input feature map every conv layer
        '''
        features = {}
        for idx,layer in enumerate(tqdm(self.model.layers)):
            if layer.name.startswith("conv"):
                feature_layer = []
                for node in layer._inbound_nodes:
                    pre_layer = node.inbound_layers
                    layer_model = keras.Model(inputs=self.model.input,
                                        outputs=pre_layer.output)
                    feature = np.array(layer_model(x))
                    feature_layer.append(feature)
                features[idx] = feature_layer
        return features

    def zero_divide(self,x,y):
        ''' Calculate x/y, deal with y=0 situation
        :param
        x   :numerator
        y   :denominator
        :return
        x/y :if 0/0 or x/0 return 1
        '''
        result = x/y
        result[np.isnan(result)] = 1
        result[np.isinf(result)] = 1
        return result

    def equalization(self,x):
        '''
        Network equalization, save the equalized model
        :param x:
        :return: None
        '''
        print("\nGenerate the original input feature map................")
        oriInfeatures = self.get_features(x)
        print("\nLayer by Layer Equalization round.................")
        for idx,layer in enumerate(tqdm(self.model.layers)):
            # For each Conv layer, get its input
            # Compare this input with the original input feature map
            if layer.name.startswith("conv"):
                oriInfeature = oriInfeatures[idx]
                scaleInChMaxs = []
                for node in layer._inbound_nodes:

                    pre_layer = node.inbound_layers
                    # print(pre_layer.name)
                    layer_model = keras.Model(inputs=self.model.input,
                                        outputs=pre_layer.output)
                    newInfeature = np.array(layer_model(x))
                    oriInChMax = np.max(oriInfeature[0], axis=(0,1,2))
                    newInChMax = np.max(newInfeature, axis=(0, 1, 2))
                    scaleInChMax = self.zero_divide(oriInChMax,newInChMax)

                    scaleInChMaxs.append(scaleInChMax)
                adj_weight = np.array(layer.get_weights())
                # print("DownScale",scaleInChMaxs[0])
                for s in range(len(scaleInChMaxs[0])):
                    adj_weight[:,:,:,s,:]*=scaleInChMaxs[0][s]
                layer.set_weights(adj_weight)

                # Calculate the kernel scale for each channel
                new_weight = np.array(layer.get_weights())
                kerOutMax = np.max(new_weight)
                kerOutChMax = np.max(new_weight,axis = (0,1,2,3))
                kerScale =  self.zero_divide(kerOutMax,kerOutChMax)

                BN_node =layer._outbound_nodes[0]
                BNlayer = BN_node.outbound_layer

                act_node = BNlayer._outbound_nodes[0]
                actlayer = act_node.outbound_layer
                actlayer_model = keras.Model(inputs=self.model.input,
                                          outputs=actlayer.output)
                # Calculate the activation scale for each channel
                act = np.array(actlayer_model(x))
                actOutMax = np.max(act)
                actOutChMax = np.max(act, axis=(0, 1, 2))
                actScale = self.zero_divide(actOutMax, actOutChMax)

                # Scale = min(actScale,kerScale,thresh)
                thresh = np.ones_like(kerScale) * self.max_thresh
                Scale = np.array([actScale, kerScale, thresh])
                Scale = np.min(Scale, axis=0)

                # Scale the kernel
                for s in range(len(Scale)):
                    new_weight[:, :, :, :, s] *= Scale[s]
                layer.set_weights(new_weight)
                # Scale the BN
                BNweight = BNlayer.get_weights()
                new_BNweight = [p * Scale for p in BNweight]
                BNlayer.set_weights(new_BNweight)

    def save_weight(self):
        self.model.save_weights(self.model_path)

def main():
    parser = argparse.ArgumentParser(description='Parameter of Network equalization.')

    parser.add_argument('--equalizedModel',default="model.h5",
                        help='save path of equalized Model')
    parser.add_argument('--scaleThresh', default=16,
                        help='scaling Thresh')
    parser.add_argument('--imagedir', default="elephant.jpg",
                        help='Image file dir for equalization')
    args = parser.parse_args()
    e_model_path = args.equalizedModel
    image_name = args.imagedir

    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)

    x = load_image(image_name)
    print("Before equalization..............................")
    preds = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', keras.applications.inception_v3.decode_predictions(preds, top=3)[0])
    equerlizer = Equalizer(model,e_model_path,args.scaleThresh)
    equerlizer.eval()
    equerlizer.equalization(x)
    model.load_weights(e_model_path)
    print("After equalization...............................")
    preds = model.predict(x)
    print('Predicted:', keras.applications.inception_v3.decode_predictions(preds, top=3)[0])


def main_equlize(model):
    parser = argparse.ArgumentParser(description='Parameter of Network equalization.')

    parser.add_argument('--equalizedModel', default="model.h5",
                        help='save path of equalized Model')
    parser.add_argument('--scaleThresh', default=16,
                        help='scaling Thresh')
    parser.add_argument('--imagedir', default="elephant.jpg",
                        help='Image file dir for equalization')
    args = parser.parse_args()
    e_model_path = args.equalizedModel

    num_data, label_to_name, val_generator = prepare_data()
    print("Before equalization..............................")
    evaluate(model)

    print("Start equalization..............................")
    equerlizer = Equalizer(model, e_model_path, args.scaleThresh)
    equerlizer.eval()
    for batch in tqdm(val_generator):
        img, label = batch
        equerlizer.get_features(img)
        equerlizer.equalization(img)

    equerlizer.save_weight()

    model.load_weights(e_model_path)
    print("After equalization...............................")
    evaluate(model)


if __name__=="__main__":
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
    main_equlize(model)