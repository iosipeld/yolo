import tensorflow as tf
from yolov3_tf2.dataset import transform_images
import time
import numpy as np
import cv2
import base64
import requests
import json
import argparse
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='0.0.0.0', help='inception serving host')
parser.add_argument('--port', default='8500', help='inception serving port')
parser.add_argument('--image', default='./data/people.jpg', help='path to JPEG image file')
FLAGS = parser.parse_args()


def main():
    # create prediction service client stub
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov3'
    request.model_spec.signature_name = 'serving_default'

    # read image into numpy array
    img = cv2.imread(FLAGS.image).astype(np.float32)
    if img is None:
        print('no image!')
    img = cv2.resize(img, (416, 416))
