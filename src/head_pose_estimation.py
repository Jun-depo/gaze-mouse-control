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

model_name = os.path.dirname("../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001")

class Model_HeadPose:
    '''
    Class for the HeadPose Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        # self.threshold = threshold

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
        print("self.model.inputs: ", self.model.inputs)
        self.input_shape=self.model.inputs[self.input_name].shape        
        self.output_name=next(iter(self.model.outputs))        
        self.output_shape=self.model.outputs[self.output_name].shape
        #print("self.input_shape: ", self.input_shape)
        #print("head_pose self.output_name:", self.output_name)
        #print("head_pose self.output_shape:", self.output_shape)

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
        if not None:
            img = self.preprocess_input(image)
            input_dict = {self.input_name: img}
            outputs = self.net.infer(input_dict)
            #print("head pose outputs", outputs)
            head_pose_angles = self.preprocess_output(outputs)
            # print("head_pose_angles shape:", head_pose_angles.shape)
            # print("head_pose_angles:", head_pose_angles)
            return head_pose_angles
        else:
            return None
        

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

        angle_y_fc = outputs["angle_y_fc"]
        angle_p_fc = outputs['angle_p_fc']
        angle_r_fc = outputs['angle_r_fc']

        if angle_y_fc or angle_p_fc or angle_r_fc:

            angle_y = np.squeeze(angle_y_fc)
            angle_p = np.squeeze(angle_p_fc)
            angle_r = np.squeeze(angle_r_fc)
            # print("angle_y:", angle_y)
            # print("angle_p:", angle_p)
            # print("angle_r:", angle_r)
            head_pose_angles = np.array([angle_y, angle_p, angle_r])  

            head_pose_angles = head_pose_angles.reshape(1, *head_pose_angles.shape) 

            return head_pose_angles

        else: 
            return None
            

