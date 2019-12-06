import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
from yolov3_tf2.dataset import transform_images

flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')


def main(_argv):
    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    transform_images(img, FLAGS.size)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
