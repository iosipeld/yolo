#!/usr/bin/env python
"""
Copyright (c) 2019, by the Authors: Amir H. Abdi
This script is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
import json
from absl import app, flags, logging
from absl.flags import FLAGS
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', False,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_string('output_nodes_prefix', None,
                    'If set, the output nodes will be renamed to '
                    '`output_nodes_prefix`+i, where `i` will numerate the '
                    'number of of output nodes of the network.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('channels_first', False,
                     'Whether channels are the first dimension of a tensor. '
                     'The default is TensorFlow behaviour where channels are '
                     'the last dimension.')
flags.DEFINE_boolean('output_meta_ckpt', False,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.mark_flag_as_required('output_model')


def main(args):
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)

    if FLAGS.channels_first:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    if FLAGS.tiny:
        model = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        model = YoloV3(classes=FLAGS.num_classes)

    model.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    # TODO(amirabdi): Support networks with multiple inputs
    orig_output_node_names = [node.op.name for node in model.outputs]
    if FLAGS.output_nodes_prefix:
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                FLAGS.output_nodes_prefix, i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names
    logging.info('Converted output node names are: %s',
                 str(converted_output_node_names))

    sess = tf.compat.v1.Session()
    if FLAGS.output_meta_ckpt:
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))

    if FLAGS.save_graph_def:
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))

    if FLAGS.quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    app.run(main)
