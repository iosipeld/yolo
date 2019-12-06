import tensorflow as tf
import time
import numpy as np
import cv2
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

    # convert to tensor proto and make request
    # shape is in NHWC (num_samples x height x width x channels) format
    #  for j in range(10):
    #   print('-- ' + str(j) + ' --')
    tt = time.time()
    tensor = tf.make_tensor_proto(img, shape=[1] + list(img.shape))
    request.inputs['input_1'].CopyFrom(tensor)
    resp = stub.Predict(request, 30.0)
    print('{}'.format((time.time() - tt) * 1000))

if __name__ == '__main__':
    main()
# img = cv2.imread('./data/people.jpg')
# img_in = tf.expand_dims(img, 0)
# img_in = transform_images(img_in, 3)
# image = np.expand_dims(img, axis=0)
# payload = {"instances": image.tolist()}
# for i in range(30):
#    t1 = time.time()
#    json_response = requests.post("http://localhost:8501/v1/models/yolov3:predict", json=payload)
#    print((time.time() - t1) * 1000)
# print(json_response.text)
