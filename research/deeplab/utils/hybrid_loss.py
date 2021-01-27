import numpy as np
import tensorflow as tf


def iou_loss(img1, img2, batch_size):
  IoU = 0.0

  for i in range(0, batch_size):
    Iand1 = tf.reduce_sum(img1[i, :, :, :] * img2[i, :, :, :])
    Ior1  = tf.reduce_sum(img1[i, :, :, :]) + tf.reduce_sum(img2[i, :, :, :]) - Iand1
    IoU1  = Iand1 / Ior1
    IoU = IoU + (1 - IoU1)

  return IoU / tf.constant(batch_size, dtype=tf.float32)


class SSIM(object):
  def __init__(self, k1=0.01, k2=0.03, window_size=11):
    self.k1 = k1
    self.k2 = k2
    self.window_size = window_size

  def gaussian(self, window_size, sigma=1.5):
    x_data, y_data = np.mgrid[-window_size//2 + 1:window_size//2 + 1, -window_size//2 + 1:window_size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    return g / tf.reduce_sum(g)

  def ssim_loss(self, img1, img2):
    window = self.gaussian(window_size=self.window_size)
    (_, _, _, channel) = img1.shape.as_list()

    window = tf.tile(window, [1, 1, channel, 1])

    mu1 = tf.nn.depthwise_conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.depthwise_conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')

    mu1_mu1 = mu1 * mu1
    mu2_mu2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    img1_img1 = img1 * img1
    sigma1_1 = tf.subtract(tf.nn.depthwise_conv2d(img1_img1, window, strides=[1, 1, 1, 1], padding='VALID'), mu1_mu1)
    img2_img2 = img2 * img2
    sigma2_2 = tf.subtract(tf.nn.depthwise_conv2d(img2_img2, window, strides=[1, 1, 1, 1], padding='VALID'), mu2_mu2)
    img1_img2 = img1 * img2
    sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img1_img2, window, strides=[1, 1, 1, 1], padding='VALID'), mu1_mu2)

    c1 = (self.k1)**2
    c2 = (self.k2)**2

    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_mu1 + mu2_mu2 + c1)*(sigma1_1 + sigma2_2 + c2))

    return tf.subtract(tf.constant(1.0, tf.float32), tf.reduce_mean(ssim_map))
