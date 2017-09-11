"""Training - mitosis detection"""
import argparse
from datetime import datetime
import os
import pickle
import shutil

import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.initializers import VarianceScaling
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from keras.models import Model
import numpy as np
import tensorflow as tf


def get_label(filename):
  """Get label from filename.

  Args:
    filename: String in format
      "**/train|val/mitosis|normal/{lab}_{case}_{region}_{row}_{col}_{suffix}.{ext}",
      where the label is either "mitosis" or "normal".

  Returns:
    TensorFlow float binary label equal to 1 for mitosis or 0 for
      normal.
  """
  # note file name format:
  # lab is a single digit, case and region are two digits with padding if needed
  # "**/train|val/mitosis|normal/{lab}_{case}_{region}_{row}_{col}_{suffix}.{ext}"
  splits = tf.string_split([filename], "/")
  label_str = splits.values[-2]
  # check that label string is valid
  is_valid = tf.logical_or(tf.equal(label_str, 'normal'), tf.equal(label_str, 'mitosis'))
  assert_op = tf.Assert(is_valid, [label_str])
  with tf.control_dependencies([assert_op]):  # test for correct label extraction
    #label = tf.to_int32(tf.equal(label_str, 'mitosis'))
    label = tf.to_float(tf.equal(label_str, 'mitosis'))  # required because model produces float
    return label


def get_image(filename, patch_size):
  """Get image from filename.

  Args:
    filename: String filename of an image.
    patch_size: Integer length to which the square image will be
      resized.

  Returns:
    TensorFlow tensor containing the decoded and resized image.
  """
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # shape (h,w,c)
  image_resized = tf.image.resize_images(image_decoded, [patch_size, patch_size])
  return image_resized


def preprocess(filename, patch_size):
  """Get image and label from filename.

  Args:
    filename: String filename of an image.
    patch_size: Integer length to which the square image will be
      resized.

  Returns:
    Tuple of a TensorFlow image tensor, a binary label, and a filename.
  """
  #  return image_resized, label
  label = get_label(filename)
  label = tf.expand_dims(label, -1)  # tf sucks
  image = get_image(filename, patch_size)
  image = normalize(image)
  return image, label, filename


def normalize(image):
  """Normalize an image.

  Args:
    image: A Tensor of shape (h,w,c).

  Returns:
    A normalized image Tensor.
  """
  image = image[..., ::-1]  # rbg -> bgr
  image = image - [103.939, 116.779, 123.68]  # mean centering using imagenet means
  return image


def create_reset_metric(metric, scope, **metric_kwargs):  # prob safer to only allow kwargs
  """Create a resettable metric.

  Args:
    metric: A tf.metrics metric function.
    scope: A String scope name to enclose the metric variables within.
    metric_kwargs:  Kwargs for the metric.

  Returns:
    The metric op, the metric update op, and a metric reset op.
  """
  # started with an implementation from https://github.com/tensorflow/tensorflow/issues/4814
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_kwargs)
    scope_name = tf.contrib.framework.get_name_scope()  # in case nested name/variable scopes
    local_vars = tf.contrib.framework.get_variables(scope_name,
        collection=tf.GraphKeys.LOCAL_VARIABLES)  # get all local variables in this scope
    reset_op = tf.variables_initializer(local_vars)
  return metric_op, update_op, reset_op


def initialize_variables(sess):
  """Initialize variables for training.

  This initializes all tensor variables in the graph, as well as
  variables for the global step and epoch, the latter two of which
  are returned as native Python variables.

  Args:
    sess: A TensorFlow Session.

  Returns:
    Integer global step and global epoch values.
  """
  # NOTE: Keras keeps track of the variables that are initialized, and any call to
  # `K.get_session()`, which is even used internally, will include a call to initialize variables
  # via `K._initalize_variables()`.  There is a situation in which resuming from a previous
  # checkpoint and then saving the model after the first epoch will result in part of the model
  # being reinitialized.  The problem is that calling `K.get_session()` here is too soon to
  # initialize any variables, and then the resume branch skips any variable initialization, and then
  # the `model.save` codepath ends up calling `K.get_session()`, thus causing part of the model to
  # be reinitialized.  Specifically, the VGG base is fine because it is initialized when the
  # pretrained weights are added in, but the new dense classifier will not be marked as initialized
  # by Keras.  The non-resume branch will initialize any variables not initialized by Keras yet, and
  # thus will avoid this issue.  It could be possible to use
  # `K.manual_variable_initialization(True)` and then manually initialize all variables, but this
  # would cause any pretrained weights to be removed.  Instead, we should initialize all variables
  # first with the equivalent of `K._initialize_variables`, and then call resume.
  # NOTE: the global variables initializer will erase the pretrained weights,
  # so we instead only initialize the other variables.
  # NOTE: reproduced from K._initialize_variables()
  # TODO: extract this out into a function and add a test case
  variables = tf.global_variables()
  uninitialized_variables = []
  for v in variables:
    if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
      uninitialized_variables.append(v)
      v._keras_initialized = True
  global_init_op = tf.variables_initializer(uninitialized_variables)
  local_init_op = tf.local_variables_initializer()
  sess.run([global_init_op, local_init_op])
  global_step = 0  # training step
  global_epoch = 0  # training epoch
  return global_step, global_epoch


def train(train_path, val_path, exp_path, patch_size, batch_size, clf_epochs, finetune_epochs,
    clf_lr, finetune_lr, finetune_layers, l2, log_interval, threads, checkpoint, resume):
  """Train a model.

  Args:
    train_path: String path to the generated training image patches.
      This should contain folders for each class.
    val_path: String path to the generated validation image patches.
      This should contain folders for each class.
    exp_path: String path in which to store the model checkpoints, logs,
      etc. for this experiment
    patch_size: Integer length to which the square patches will be
      resized.
    batch_size: Integer batch size.
    clf_epochs: Integer number of epochs for which to training the new
      classifier layers.
    finetune_epochs: Integer number of epochs for which to fine-tune the
      model.
    clf_lr: Float learning rate for training the new classifier layers.
    finetune_lr: Float learning rate for fine-tuning the model.
    finetune_layers: Integer number of layers at the end of the
      pretrained portion of the model to fine-tune.  The new classifier
      layers will still be trained during fine-tuning as well.
    l2: Float L2 global regularization value.
    log_interval: Integer number of steps between logging during
      training.
    threads: Integer number of threads for dataset buffering.
    checkpoint: Boolean flag for whether or not to save a checkpoint
      after each epoch.
    resume: Boolean flag for whether or not to resume training from a
      checkpoint.
  """
  # TODO: break this out into:
  #   * data gen func
  #   * inference func
  #   * loss func
  #   * metrics func
  #   * logging func
  #   * train func

  sess = K.get_session()

  # debugger
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  # data
  with tf.name_scope("data"):
    # TODO: add data augmentation function
    train_dataset = (tf.contrib.data.Dataset.list_files('{}/*/*.jpg'.format(train_path))
        .shuffle(10000)
        .map(lambda x: preprocess(x, patch_size), num_threads=threads,
          output_buffer_size=100*batch_size)
        .batch(batch_size)
        )
    val_dataset = (tf.contrib.data.Dataset.list_files('{}/*/*.jpg'.format(val_path))
        .map(lambda x: preprocess(x, patch_size), num_threads=threads,
          output_buffer_size=100*batch_size)
        .batch(batch_size)
        )

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
    images, labels, filenames = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)
    actual_batch_size = tf.shape(images)[0]
    input_shape = (patch_size, patch_size, 3)

  # models
  with tf.name_scope("model"):
    ## logistic regression classifier
    #model_base = keras.models.Sequential()  # dummy since we aren't fine-tuning this model
    #inputs = Input(shape=input_shape)
    #x = Flatten()(inputs)
    ## init Dense weights with Gaussian scaled by sqrt(2/(fan_in+fan_out))
    #logits = Dense(1, kernel_initializer="glorot_normal",
    #    kernel_regularizer=keras.regularizers.l2(l2))(x)
    #model_tower = Model(inputs=inputs, outputs=logits, name="model")

    # Create model by replacing classifier of VGG16 model with new
    # classifier specific to the breast cancer problem.
    #with tf.device("/cpu"):
    inputs = Input(shape=input_shape)
    model_base = VGG16(include_top=False, input_shape=input_shape, input_tensor=inputs)
    x = model_base.output
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    # init Dense weights with Gaussian scaled by sqrt(2/(fan_in+fan_out))
    logits = Dense(1, kernel_initializer="glorot_normal",
        kernel_regularizer=keras.regularizers.l2(l2))(x)
    model_tower = Model(inputs=inputs, outputs=logits, name="model")

    ## Create model by replacing classifier of ResNet50 model with new
    ## classifier specific to the breast cancer problem.
    ##with tf.device("/cpu"):
    #inputs = Input(shape=input_shape)
    #model_base = ResNet50(include_top=False, input_shape=input_shape, input_tensor=inputs)
    #x = model_base.output
    #x = Flatten()(x)
    ##x = GlobalAveragePooling2D()(x)
    ## init Dense weights with Gaussian scaled by sqrt(2/(fan_in+fan_out))
    #logits = Dense(1, kernel_initializer="glorot_normal",
    #    kernel_regularizer=keras.regularizers.l2(l2))(x)
    #model_tower = Model(inputs=inputs, outputs=logits, name="model")

    # TODO: add this when it's necessary, and move to a separate function
    ## Multi-GPU exploitation via a linear combination of GPU loss functions.
    #ins = []
    #outs = []
    #for i in range(num_gpus):
    #  with tf.device("/gpu:{}".format(i)):
    #    x = Input(shape=input_shape)  # split of batch
    #    out = resnet50(x)  # run split on shared model
    #    ins.append(x)
    #    outs.append(out)
    #model = Model(inputs=ins, outputs=outs)  # multi-GPU, data-parallel model
    model = model_tower
    #import pdb; pdb.set_trace()

    # call model on dataset images to compute logits and predictions
    logits = model(images)  # tf prefers to feed logits into a combined sigmoid and logistic loss function for numerical stability
    preds = tf.round(tf.nn.sigmoid(logits), name="preds")  # for metric computation; implicit threshold at 0.5

  # loss
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

  # optim
  with tf.name_scope("optim"):
    # classifier
    # - freeze all pre-trained model layers.
    for layer in model_base.layers:
      layer.trainable = False
    clf_opt = tf.train.AdamOptimizer(clf_lr)
    clf_grads_and_vars = clf_opt.compute_gradients(loss, var_list=model.trainable_weights)
    #clf_train_op = opt.minimize(loss, var_list=model.trainable_weights)
    clf_apply_grads_op = clf_opt.apply_gradients(clf_grads_and_vars)
    clf_model_update_ops = model.updates
    clf_train_op = tf.group(clf_apply_grads_op, *clf_model_update_ops)

    # finetuning
    # - unfreeze a portion of the pre-trained model layers.
    # note, could make this arbitrary, but for now, fine-tune some number of layers at the *end* of
    # the pretrained portion of the model
    for layer in model_base.layers[-finetune_layers:]:
      layer.trainable = True
    finetune_opt = tf.train.AdamOptimizer(finetune_lr)
    finetune_grads_and_vars = finetune_opt.compute_gradients(loss, var_list=model.trainable_weights)
    #finetune_train_op = opt.minimize(loss, var_list=model.trainable_weights)
    finetune_apply_grads_op = finetune_opt.apply_gradients(finetune_grads_and_vars)
    finetune_model_update_ops = model.updates
    finetune_train_op = tf.group(finetune_apply_grads_op, *finetune_model_update_ops)

  # metrics
  with tf.name_scope("metrics"):
    mean_loss, mean_loss_update_op, mean_loss_reset_op = create_reset_metric(tf.metrics.mean,
        'mean_loss', values=loss)
    acc, acc_update_op, acc_reset_op = create_reset_metric(tf.metrics.accuracy, 'acc',
        labels=labels, predictions=preds)
    ppv, ppv_update_op, ppv_reset_op = create_reset_metric(tf.metrics.precision,
        'ppv', labels=labels, predictions=preds)
    recall, recall_update_op, recall_reset_op = create_reset_metric(tf.metrics.recall,
        'recall', labels=labels, predictions=preds)
    f1 = 2 * (ppv * recall) / (ppv + recall)

    # combine all reset & update ops
    metric_update_ops = tf.group(mean_loss_update_op, acc_update_op, ppv_update_op,
        recall_update_op)
    metric_reset_ops = tf.group(mean_loss_reset_op, acc_reset_op, ppv_reset_op,
        recall_reset_op)

  # tensorboard
  #with tf.name_scope("logging"):
  # minibatch summaries
  images_summary = tf.summary.image("images", images) #, max_outputs=10)
  actual_batch_size_summary = tf.summary.scalar("batch_size", actual_batch_size)
  minibatch_loss_summary = tf.summary.scalar("minibatch_loss", loss)
  minibatch_summaries = tf.summary.merge([minibatch_loss_summary]) #, actual_batch_size_summary,
      #images_summary])
  # epoch summaries
  epoch_loss_summary = tf.summary.scalar("epoch_avg_loss", mean_loss)
  epoch_acc_summary = tf.summary.scalar("epoch_acc", acc)
  epoch_ppv_summary = tf.summary.scalar("epoch_ppv", ppv)
  epoch_recall_summary = tf.summary.scalar("epoch_recall", recall)
  epoch_f1_summary = tf.summary.scalar("epoch_f1", f1)
  epoch_summaries = tf.summary.merge([epoch_loss_summary, epoch_acc_summary,
    epoch_ppv_summary, epoch_recall_summary, epoch_f1_summary])
  #all_summaries = tf.summary.merge_all()

  # use train and val writers so that plots can be on same graph
  writer = tf.summary.FileWriter(exp_path, sess.graph)
  train_writer = tf.summary.FileWriter(os.path.join(exp_path, "train"))
  val_writer = tf.summary.FileWriter(os.path.join(exp_path, "val"))

  # save ops
  checkpoint_filename = os.path.join(exp_path, "model.ckpt")
  global_step_epoch_filename = os.path.join(exp_path, "global_step_epoch.pickle")
  saver = tf.train.Saver()

  # initialize stuff
  global_step, global_epoch = initialize_variables(sess)

  if resume:
    saver.restore(sess, checkpoint_filename)
    with open(global_step_epoch_filename, "rb") as f:
      global_step, global_epoch = pickle.load(f)

  # new classifier layers + fine-tuning combined training loop
  for train_op, epochs in [(clf_train_op, clf_epochs), (finetune_train_op, finetune_epochs)]:
    for _ in range(global_epoch, global_epoch+epochs):  # allow for resuming of training
      # training
      sess.run(train_init_op)
      while True:
        try:
          if log_interval > 0 and global_step % log_interval == 0:
            # train, update metrics, & log stuff
            _, _, loss_val, summary_str, mean_loss_val, acc_val = sess.run([train_op,
                metric_update_ops, loss, minibatch_summaries, mean_loss, acc],
                feed_dict={K.learning_phase(): 1})
            train_writer.add_summary(summary_str, global_step)
            print("train", global_epoch, global_step, loss_val, mean_loss_val, acc_val)
          else:
            # train & update metrics
            _, _, loss_val = sess.run([train_op, metric_update_ops, minibatch_loss_summary],
                feed_dict={K.learning_phase(): 1})
          global_step += 1
        except tf.errors.OutOfRangeError:
          break
      # log average training metrics for epoch & reset
      print("---epoch {}, train avg loss: {}, train acc: {}".format(global_epoch,
          *sess.run([mean_loss, acc])))
      train_writer.add_summary(sess.run(epoch_summaries), global_epoch)
      sess.run(metric_reset_ops)

      # validation
      sess.run(val_init_op)
      vi = 0  # validation step
      while True:
        try:
          # evaluate & update metrics
          _, loss_val, mean_loss_val, acc_val = sess.run([metric_update_ops, loss, mean_loss, acc],
              feed_dict={K.learning_phase(): 0})
          if log_interval > 0 and vi % log_interval == 0:
            print("val", global_epoch, vi, loss_val, mean_loss_val, acc_val)
          vi += 1
        except tf.errors.OutOfRangeError:
          break
      # log average validation metrics for epoch & reset
      loss_val, acc_val = sess.run([mean_loss, acc])
      print("---epoch {}, val avg loss: {}, val acc: {}".format(global_epoch, loss_val, acc_val))
      val_writer.add_summary(sess.run(epoch_summaries), global_epoch)
      sess.run(metric_reset_ops)

      val_writer.flush()
      #train_writer.flush()

      global_epoch += 1

      # save model
      if checkpoint:
        # TODO: save model with sigmoid function appended
        keras_filename = os.path.join(exp_path,
            f"{acc_val:.5}_acc_{loss_val:.5}_loss_{global_epoch}_epoch_model.hdf5")
        model.save(keras_filename, include_optimizer=False)  # keras model
        saver.save(sess, checkpoint_filename)  # full TF graph
        with open(global_step_epoch_filename, "wb") as f:
          pickle.dump((global_step, global_epoch), f)  # step & epoch
        print("Saved model file to {}".format(keras_filename))


if __name__ == "__main__":
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--patches_path", default=os.path.join("data", "mitoses", "patches"),
      help="path to the generated image patches containing `train` & `val` folders "\
           "(default: %(default)s)")
  parser.add_argument("--exp_parent_path", default=os.path.join("experiments", "mitoses", "sanity"),
      help="parent path in which to store experiment folders (default: %(default)s)")
  parser.add_argument("--exp_name", default=None,
      help="path within the experiment parent path in which to store the model checkpoints, "\
           "logs, etc. for this experiment "\
           "(default: %%y-%%m-%%d_%%H:%%M:%%S_patch_size=x_batch_size=x_clf_epochs=x_"\
           "finetune_epochs=x_clf_lr=x_finetune_lr=x_l2=x)")
  parser.add_argument("--patch_size", type=int, default=64,
      help="integer length to which the square patches will be resized (default: %(default)s)")
  parser.add_argument("--batch_size", type=int, default=32,
      help="batch size (default: %(default)s)")
  parser.add_argument("--clf_epochs", type=int, default=1,
      help="number of epochs for which to train the new classifier layers "\
           "(default: %(default)s)")
  parser.add_argument("--finetune_epochs", type=int, default=0,
      help="number of epochs for which to fine-tune the unfrozen layers (default: %(default)s)")
  parser.add_argument("--clf_lr", type=float, default=1e-5,
      help="learning rate for training the new classifier layers (default: %(default)s)")
  parser.add_argument("--finetune_lr", type=float, default=1e-7,
      help="learning rate for fine-tuning the unfrozen layers (default: %(default)s)")
  parser.add_argument("--finetune_layers", type=int, default=0,
      help="number of layers at the end of the pretrained portion of the model to fine-tune "\
          "(note: the new classifier layers will still be trained during fine-tuning as well) "\
          "(default: %(default)s)")
  parser.add_argument("--l2", type=float, default=0.01,
      help="amount of l2 weight regularization (default: %(default)s)")
  parser.add_argument("--log_interval", type=int, default=100,
      help="number of steps between logging during training (default: %(default)s)")
  parser.add_argument("--threads", type=int, default=5,
      help="number of threads for dataset buffering (default: %(default)s)")
  parser.add_argument("--resume", default=False, action="store_true",
      help="resume training from a checkpoint (default: %(default)s)")
  checkpoint_parser = parser.add_mutually_exclusive_group(required=False)
  checkpoint_parser.add_argument("--checkpoint", dest="checkpoint", action="store_true",
      help="save a checkpoint after each epoch (default: True)")
  checkpoint_parser.add_argument("--no_checkpoint", dest="checkpoint", action="store_false",
      help="do not save a checkpoint after each epoch (default: False)")
  parser.set_defaults(checkpoint=True)

  args = parser.parse_args()

  # set any other defaults
  train_path = os.path.join(args.patches_path, "train")
  val_path = os.path.join(args.patches_path, "val")

  if args.exp_name == None:
    date = datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
    args.exp_name = f"{date}_patch_size_{args.patch_size}_batch_size_{args.batch_size}_"\
                    f"clf_epochs_{args.clf_epochs}_finetune_epochs_{args.finetune_epochs}_"\
                    f"clf_lr_{args.clf_lr}_finetune_lr_{args.finetune_lr}_l2_{args.l2}"
  exp_path = os.path.join(args.exp_parent_path, args.exp_name)

  # make experiment folder
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  print("experiment directory: {}".format(exp_path))

  # save args to file in experiment folder
  with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
    f.write(str(args))

  # copy this script to the base save folder
  shutil.copy2(os.path.realpath(__file__), base_save_path)

  # train!
  train(train_path, val_path, exp_path, args.patch_size, args.batch_size,
      args.clf_epochs, args.finetune_epochs, args.clf_lr, args.finetune_lr, args.finetune_layers,
      args.l2, args.log_interval, args.threads, args.checkpoint, args.resume)


# ---
# tests
# TODO: eventually move these to a separate file.
# `py.test train_mitoses.py`

def test_get_label():
  import pytest
  sess = tf.Session()

  # mitosis
  filename = "train/mitosis/1_03_05_713_348.jpg"
  label_op = get_label(filename)
  label = sess.run(label_op)
  assert label == 1

  # normal
  filename = "train/normal/1_03_05_713_348.jpg"
  label_op = get_label(filename)
  label = sess.run(label_op)
  assert label == 0

  # wrong label name
  with pytest.raises(tf.errors.InvalidArgumentError):
    filename = "train/unknown/1_03_05_713_348.jpg"
    label_op = get_label(filename)
    label = sess.run(label_op)


def test_resettable_metric():
  x = tf.placeholder(tf.int32, [None, 1])
  x1 = np.array([1,0,0,0]).reshape(4,1)
  x2 = np.array([0,0,0,0]).reshape(4,1)

  with tf.name_scope("something"):  # testing nested name/variable scopes
    mean_op, update_op, reset_op = create_reset_metric(tf.metrics.mean, 'mean_loss', values=x)

  sess = K.get_session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [1/4])
  assert np.allclose([sess.run(mean_op)], [1/4])
  assert np.allclose([sess.run(mean_op)], [1/4])

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [2/8])
  assert np.allclose([sess.run(mean_op)], [2/8])
  assert np.allclose([sess.run(mean_op)], [2/8])

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x2})
  assert np.allclose([out_up], [2/12])
  assert np.allclose([sess.run(mean_op)], [2/12])
  assert np.allclose([sess.run(mean_op)], [2/12])

  sess.run(reset_op)  # make sure this works!

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x2})
  assert out_up == 0
  assert sess.run(mean_op) == 0

  _, out_up = sess.run([mean_op, update_op], feed_dict={x: x1})
  assert np.allclose([out_up], [1/8])
  assert np.allclose([sess.run(mean_op)], [1/8])
  assert np.allclose([sess.run(mean_op)], [1/8])


def test_initialize_variables():
  import pytest
  sess = K.get_session()

  # create model with a mix of pretrained and new weights
  # NOTE: the pretrained layers will be initialized by Keras on creation, while the new Dense
  # layer will remain uninitialized
  input_shape = (224,224,3)
  inputs = Input(shape=input_shape)
  model_base = VGG16(include_top=False, input_shape=input_shape, input_tensor=inputs)
  x = model_base.output
  x = GlobalAveragePooling2D()(x)
  logits = Dense(1)(x)
  model = Model(inputs=inputs, outputs=logits, name="model")

  # the new dense layer is not initialized yet
  with pytest.raises(AssertionError):
    for v in tf.global_variables():
      assert hasattr(v, '_keras_initialized') and v._keras_initialized  # check for initialization

  # initialize variables, including marking them with the `_keras_initialized` attribute
  initialize_variables(sess)

  # check that everything is initialized and marked with the `_keras_initialized` attribute
  # NOTE: this is important for a hybrid Keras & TensorFlow setup where Keras is being used for the
  # model creation part, and raw TensorFlow is being used for the rest.  if variables are not
  # initialized *and* marked with the special Keras attribute, then certain Keras functions will end
  # up accidentally reinitializing variables when they use `K.get_session()` internally.  In a pure
  # Keras setup, this would not happen since the model would be initialized at the proper times.  In
  # a Keras & TensorFlow hybrid setup, this can cause issues.  By encapsulating this nonsense in a
  # function, we can avoid these problems.
  for v in tf.global_variables():
    assert hasattr(v, '_keras_initialized') and v._keras_initialized  # check for initialization

