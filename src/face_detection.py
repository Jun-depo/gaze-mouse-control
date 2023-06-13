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

class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        model_weights=self.model_name+'.bin'
        model_structure=self.model_name+'.xml'
        try:
            self.model=IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the IENetwork. Have you enterred the correct model path?")

        self.plugin = IECore()
        if self.extensions:
            self.plugin.add_extension(extension_path=self.extensions, device_name=self.device)

        self.input_name=next(iter(self.model.inputs))
        #print("self.input_name: ", self.input_name)
        self.input_shape=self.model.inputs[self.input_name].shape
        print("self.input_shape: ", self.input_shape)
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''         
        unsupported_layers = self.check_model()        
        if len(unsupported_layers) < 1:
            self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests = 1)
            
        else:                        
            print("found following unsupported layers:", unsupported_layers)

        return self.net

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img = self.preprocess_input(image)
        input_dict = {self.input_name: img}
        outputs = self.net.infer(input_dict)
        boxes = self.preprocess_output(outputs)
        if len(boxes)==0:
            return [], image
        
        else:
            return boxes

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img = self.preprocess_input(image)
        input_dict = {self.input_name: img}
        outputs = self.net.infer(input_dict)
        boxes = self.preprocess_output(outputs)
        if len(boxes)==0:
            return [], image
        
        else:
            return boxes

    # def predict_async(self, image, request_id=0):
        
    #     img = self.preprocess_input(image)
    #     input_dict = {self.input_name: img}
    #     outputs = self.net.start_async(request_id, inputs=input_dict)
    #     boxes = self.preprocess_output(outputs)
    #     if len(boxes)==0:
    #         return [], image
        
    #     else:
    #         return boxes


    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)

        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        return unsupported_layers

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''          
        img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))        
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape) 
        return img

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # outputs blob with shape: [1, 1, N, 7], N number of bounding boxes
        # Each box [image_id, label, conf, x_min, y_min, x_max, y_max]

        outputs = outputs['detection_out']
        if len(outputs)==0:
            return []
        else:
            boxes = np.squeeze(outputs)
        
            boxes = boxes[(boxes[:, 1] == 1) & (boxes[:, 2] > float(self.threshold))]

        if len(boxes)==0:
            return []
        else:
            return boxes


