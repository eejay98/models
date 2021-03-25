"""
Visualize inference results.
"""

import os
import glob
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string('export_dir', None, help='The exported model directory.')
flags.DEFINE_string(
    'input_dir', None, help='A single JPEG image file or the directory '
    'with multiple JPEG image files')
flags.DEFINE_string(
    'save_dir', None, help='The directory where the inference results are saved.')
flags.DEFINE_string('task', default='semantic_segmentation', help='the name of task.')


def main(_):
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  # read image files.
  file_list = []
  if os.path.isdir(FLAGS.input_dir):
    file_list = glob.glob(os.path.join(FLAGS.input_dir, "*.jpg"))
  else:
    file_list.append(FLAGS.input_dir)

  # generate pascal color map.
  colormap = np.zeros((512, 3), dtype=int)
  ind = np.arange(512, dtype=int)

  for shift in reversed(list(range(8))):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  # load exported model.
  imported = tf.saved_model.load(FLAGS.export_dir)
  model_fn = imported.signatures['serving_default']

  # visualize input files.
  for i, file_name in enumerate(file_list):
    images = np.array(Image.open(file_name))
    (height, width, channel) = images.shape

    # basnet
    if FLAGS.task == 'basnet':
      images = tf.image.resize(images, [256, 256])
      images = tf.expand_dims(images, axis=0)
      images = tf.cast(images, tf.uint8)

      outputs = model_fn(images)['outputs']

      outputs = outputs * 255
      outputs = tf.image.resize(tf.cast(outputs, tf.uint8), [height, width])
      outputs = tf.image.grayscale_to_rgb(outputs)
      outputs = tf.squeeze(outputs, axis=0)

    # deeplab v3+
    else:
      images = tf.image.resize(images, [512, 512])
      images = tf.expand_dims(images, axis=0)
      images = tf.cast(images, tf.uint8)

      outputs = model_fn(images)['predicted_masks']

      outputs = tf.argmax(tf.squeeze(outputs, axis=0), axis=-1)
      outputs = tf.expand_dims(outputs, axis=-1)

    outputs = outputs.numpy()

    if FLAGS.task == 'semantic_segmentation':
      # color mapping.
      colored_label = np.zeros((512, 512, 3), dtype=int)
      for x in range(512):
        for y in range(512):
          colored_label[x][y][:] = colormap[int(outputs[x][y][0])][:]
      # TODO: determine when to resize the image.
      outputs = tf.convert_to_tensor(colored_label, dtype=tf.uint8)
      outputs = tf.image.resize(outputs, [height, width])
      outputs = outputs.numpy()

    convert_to_png = os.path.join(FLAGS.save_dir, os.path.basename(file_name).replace(".jpg", ".png"))
    tf.keras.preprocessing.image.save_img(
        convert_to_png,
        outputs,
        data_format="channels_last",
        scale=False)
    
    if i % 100 == 0:
      print("progress : " + str(i) + " of " + str(len(file_list)))


if __name__ == '__main__':
  app.run(main)