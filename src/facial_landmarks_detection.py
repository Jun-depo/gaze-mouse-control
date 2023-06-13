'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore

import os
import cv2
import argparse
import sys

from input_feeder import InputFeeder

#model_name = os.path.dirname("../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin")

class Model_FaceLandmark:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions

        model_weights=self.model_name+'.bin'
        model_structure=self.model_name+'.xml'
        try:
            self.model=IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the IENetwork. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        print("FaceLandmark self.input_shape: ", self.input_shape)
        print("FaceLandmark self.output_name:", self.output_name)
        print("FaceLandmark self.output_shape:", self.output_shape)

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        
        self.plugin = IECore()
        if self.extensions:
            self.plugin.add_extension(extension_path=self.extensions, device_name=self.device)
        
        unsupported_layers = self.check_model()        
        if len(unsupported_layers) < 1:
            self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests = 1)
            return self.net
        else:                        
            print("found following unsupported layers:", unsupported_layers)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img = self.preprocess_input(image)
        input_dict = {self.input_name: img}
        outputs = self.net.infer(input_dict)
        
        landmark_outputs = self.preprocess_output(outputs)
        #print("Face_landmark outputs", outputs)
        #print("landmark output shape after preprocessing", landmark_outputs.shape)
        return landmark_outputs

    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)

        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        return unsupported_layers

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        img = np.copy(image)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))       
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape) 
        return img

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs=outputs['95']
        landmark_outputs = np.squeeze(outputs) 
        
        return landmark_outputs