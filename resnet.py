"""Custom ResNet model with pre-activation residual blocks.

He K, Zhang X, Ren S, Sun J. Identity Mappings in Deep Residual
Networks. arXiv.org. 2016.

Author: Mike Dusenberry
"""
import tensorflow as tf


def res_block(xin, dbottle, dout, k, stride):
  """A residual block.

  This implements the "pre-activation" formulation of a residual block,
  as discussed in:

    He K, Zhang X, Ren S, Sun J. Identity Mappings in Deep Residual
    Networks. arXiv.org. 2016.

  Args:
    xin: Input tensor.
    dbottle: Bottleneck depth.
    dout: Output depth.
    k: Integer kernel size.
    stride: Integer stride.

  Returns:
    Output tensor for the block.
  """
  depth_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
  din = tf.shape(xin)[depth_axis]  # input depth
  he_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')

  # TODO: ReLUs have been quite successful, but it still seems like it could be a problem due to
  # gradient stopping at ReLU zero values.  Perhaps look into leaky ReLUs, ELUs, etc.

  # conv 1x1
  x = tf.keras.layers.BatchNormalization(axis=depth_axis, momentum=0.9, epsilon=1e-4)(xin)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Conv2D(
      dbottle, (1, 1), strides=(stride, stride), kernel_initializer=he_init)(x)

  # conv 3x3
  x = tf.keras.layers.BatchNormalization(axis=depth_axis, momentum=0.9, epsilon=1e-4)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Conv2D(dbottle, (k, k), padding='same', kernel_initializer=he_init)(x)

  # conv 1x1
  x = tf.keras.layers.BatchNormalization(axis=depth_axis, momentum=0.9, epsilon=1e-4)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Conv2D(dout, (1, 1), kernel_initializer=he_init)(x)

  # shortcut
  if din == dout:  # identity shortcut for same input/output depths
    shortcut = xin
  else:  # conv shortcut to change depth (usually to increase depth)
    shortcut = tf.keras.layers.Conv2D(
        dout, (1, 1), strides=(stride, stride), kernel_initializer=he_init)(xin)

  x = tf.keras.layers.add([x, shortcut])

  return x


def ResNet(xin, shape):  # camel case makes it feel like a class -- eventually we'll subclass Model
  """Custom ResNet model with pre-activation residual blocks.

  Reference:

    He K, Zhang X, Ren S, Sun J. Identity Mappings in Deep Residual
    Networks. arXiv.org. 2016.

  Args:
    xin: Input tensor.
    shape: Integer tuple of length 3 containing the shape of a single
      example.

  Returns:
    A Keras Model.

  Example:
    ```
    import tensorflow as tf
    import numpy as np
    import resnet

    shape = (64, 64, 3)
    xin = tf.placeholder(tf.float32, shape=(None, *shape))
    model = resnet.ResNet(xin, shape)

    model.summary()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run(model.output, feed_dict={xin: np.random.randn(10, *shape)})
    print(out)
    ```
  """
  # TODO: `tf.keras.layers` -> `tf.layers`
  assert len(shape) == 3
  depth_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

  d = [16, 32, 64, 128]  # depths (must be divisible by 4)
  db = [int(depth/4) for depth in d]  # bottleneck depths
  n = 3  # num layers at each depth

  # input & conv
  with tf.variable_scope("beg"):
    xin = tf.keras.layers.Input(tensor=xin, shape=shape)  # shape (h,w,c)
    he_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
    x = tf.keras.layers.Conv2D(
        d[0], (3, 3), strides=(2, 2),
        padding='same', kernel_initializer=he_init)(xin)  # shape (h/2,w/2,d[0])

  # stage 1
  with tf.variable_scope("stage1"):
    x = res_block(x, db[0], d[1], 3, 1)  # shape (h/2,w/2,d[1]) <-- increase depth
    for i in range(n-1):
      x = res_block(x, db[1], d[1], 3, 1)  # shape (h/2,w/2,d[1])

  # stage 2
  with tf.variable_scope("stage2"):
    x = res_block(x, db[1], d[2], 3, 2)  # shape (h/4,w/4,d[2]) <-- increase depth, cut spatial size
    for i in range(n-1):
      x = res_block(x, db[2], d[2], 3, 1)  # shape (h/4,w/4,d[2])

  # stage 3
  with tf.variable_scope("stage3"):
    x = res_block(x, db[2], d[3], 3, 2)  # shape (h/8,w/8,d[3]) <-- increase depth, cut spatial size
    for i in range(n-1):
      x = res_block(x, db[3], d[3], 3, 1)  # shape (h/8,w/8,d[3])

  # final functions
  with tf.variable_scope("end"):
    x = tf.keras.layers.BatchNormalization(
        axis=depth_axis, momentum=0.9, epsilon=1e-4)(x)  # shape (h/8,w/8,d[3])
    x = tf.keras.layers.Activation('relu')(x)  # shape (h/8,w/8,d[3])
    x = tf.keras.layers.AvgPool2D((8, 8))(x)  # shape (h/64,w/64,d[3])
    init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
    # TODO: this is a binary classification problem so optimizing a loss derived from a Bernoulli
    # distribution is appropriate.  however, would the dynamics of the training algorithm be more
    # stable if we treated this as a multi-class classification problem and derived a loss from a
    # Multinomial distribution with two classes (and a single trial)?  it would be
    # over-parameterized, but then again, the deep net itself is already heavily parameterized.
    x = tf.keras.layers.Conv2D(
        1, (1, 1), kernel_initializer=init)(x)  # shape (h/64,w/64,1) <-- could use this for surgery
    x = tf.keras.layers.Flatten()(x)  # shape ((h/64)*(w/64)*1)  <-- normally will be a single value

  # create model (106 functions)
  model = tf.keras.Model(xin, x, name='resnet')

  return model

