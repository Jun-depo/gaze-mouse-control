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


class Model_GazeEstimation:
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

        #self.input_head_pose=next(iter(self.model.inputs['head_pose_angles']))
        self.input_left_eye = 'left_eye_image'
        self.input_right_eye = 'right_eye_image'
        self.input_head_pose = 'head_pose_angles'

        self.left_input_shape=self.model.inputs['left_eye_image'].shape
        self.right_input_shape=self.model.inputs['right_eye_image'].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


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


    def predict_async(self, img_left, img_right, head_pose_angles, request_id=0):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img_left, img_right, head_pose_angles = self.preprocess_input(img_left, img_right, head_pose_angles)
        input_dict = {'left_eye_image': img_left, 'right_eye_image': img_right, 'head_pose_angles': head_pose_angles}
        self.net.start_async(request_id, inputs=input_dict)
        self.net.requests[0].wait(-1)
        outputs = self.net.requests[0].outputs[self.output_name]
        gaze_vector = self.preprocess_output(outputs)  # Cartesian coordinates (x, y, z)
        return gaze_vector

    def check_model(self):
        
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        return unsupported_layers

    def preprocess_input(self, img_left, img_right, head_pose_angles):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        '''

        img_left = np.copy(img_left)
        img_left = cv2.resize(img_left, (self.left_input_shape[3], self.left_input_shape[2]))        
        img_left = img_left.transpose((2,0,1))
        img_left = img_left.reshape(1, *img_left.shape) 

        
        img_right = np.copy(img_right)
        img_right = cv2.resize(img_right, (self.right_input_shape[3], self.right_input_shape[2]))        
        img_right = img_right.transpose((2,0,1))
        img_right = img_right.reshape(1, *img_right.shape)

        return img_left, img_right, head_pose_angles

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # gaze_vector = outputs['gaze_vector']
        gaze_vector = np.squeeze(outputs)

        return gaze_vector
