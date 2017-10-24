"""Evaluation - mitosis detection"""
import argparse
import os

from keras.models import load_model
import keras.backend as K
import numpy as np
import tensorflow as tf

from train_mitoses import create_dataset, compute_data_loss, compute_metrics, marginalize


def evaluate(patches_path, patch_size, batch_size, model, model_name, prob_threshold,
    marginalization, threads, prefetch_batches, log_interval):
  """Evaluate a model.

  Args:
    patches_path: String path to the generated image patches.  This
      should contain folders for each class.
    patch_size: Integer length to which the square patches will be
      resized.
    batch_size: Integer batch size.
    model: A Keras Model object.
    model_name: String indicating the model to use.
    prob_threshold: Decimal threshold over which the patch is predicted as a
      positive case.
    marginalization: Boolean for whether or not to use noise
      marginalization when evaluating the validation set.  If True, then
      each image in the validation set will be expanded to a batch of
      augmented versions of that image, and predicted probabilities for
      each batch will be averaged to yield a single noise-marginalized
      prediction for each image.  Note: if this is True, then
      `val_batch_size` must be divisible by 4, or equal to 1 for a
      special debugging case of no augmentation.
    threads: Integer number of threads for dataset buffering.
    prefetch_batches: Integer number of batches to prefetch.
    log_interval: Integer number of steps between logging during
      training.

  Returns:
    F1 score, ppv (precision), sensitivity (recall), accuracy, and loss
    values on the dataset.
  """
  # data
  with tf.name_scope("data"):
    dataset = create_dataset(patches_path, model_name, patch_size, batch_size, False, False,
        marginalization, threads, prefetch_batches)

    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    data_init_op = iterator.make_initializer(dataset)
    images, labels, filenames = iterator.get_next()
    input_shape = (patch_size, patch_size, 3)

    if marginalization:
      labels = marginalize(labels)  # will marginalize at test time

  # models
  with tf.name_scope("model"):
    # compute logits and predictions, possibly with marginalization
    # NOTE: tf prefers to feed logits into a combined sigmoid and logistic loss function for
    # numerical stability
    if marginalization:
      logits = marginalize(model(images))  # will marginalize at test time
    else:
      logits = model(images)
    probs = tf.nn.sigmoid(logits, name="probs")
    #preds = tf.round(probs, name="preds")  # implicit threshold at 0.5
    preds = probs > prob_threshold

  # loss
  with tf.name_scope("loss"):
    # NOTE: for now, we ignore regularization.  loss at evaluation time is fairly useless anyway...
    data_loss = compute_data_loss(labels, logits)
    loss = data_loss

  # metrics
  with tf.name_scope("metrics"):
    mean_loss, acc, ppv, sens, f1, metric_update_ops, metric_reset_ops = compute_metrics(loss,
        labels, preds)

  # initialize stuff
  sess = K.get_session()
  sess.run([data_init_op, metric_reset_ops])

  # evaluation
  vi = 0  # validation step
  while True:
    try:
      # evaluate & update metrics
      sess.run(metric_update_ops, feed_dict={K.learning_phase(): 0})

      if log_interval > 0 and vi % log_interval == 0:
        metrics = sess.run([f1, ppv, sens, acc, mean_loss])
        f1_val, ppv_val, sens_val, acc_val, mean_loss_val = metrics
        print("val", vi, f1_val, ppv_val, sens_val, acc_val, mean_loss_val)

      vi += 1
    except tf.errors.OutOfRangeError:
      break

  # log average validation metrics
  f1_val, ppv_val, sens_val, acc_val, mean_loss_val = sess.run([f1, ppv, sens, acc, mean_loss])
  print("f1: {}".format(f1_val))
  print("ppv: {}".format(ppv_val))
  print("sens: {}".format(sens_val))
  print("acc: {}".format(acc_val))
  print("loss: {}".format(mean_loss_val))

  return f1_val, ppv_val, sens_val, acc_val, mean_loss_val


def main(args=None):
  """Command line interface for this script.  Can optionally pass in a
  list of strings to simulate command line usage.
  """
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--patches_path",
      help="path to the generated image patches containing `mitosis` & `normal` folders")
  parser.add_argument("--patch_size", type=int, default=64,
      help="integer length to which the square patches will be resized (default: %(default)s)")
  parser.add_argument("--batch_size", type=int, default=32,
      help="batch size (default: %(default)s)")
  parser.add_argument("--model_name",
      help="name of the model being used in ['logreg', 'vgg', 'vgg19', 'resnet'], which is used "\
           "for determining the correct normalization")
  parser.add_argument("--model_path",
      help="path to a Keras model to use for false-positive oversampling; note: this model should "\
           "produce logit values, rather than probability values")
  parser.add_argument("--prob_threshold", type=float, default=0.5,
      help="threshold over which the patch is predicted as a positive case (default: %(default)s)")
  parser.add_argument("--marginalize", default=False, action="store_true",
      help="use noise marginalization when evaluating the validation set. if this is set, then "\
           "the validation batch_size must be divisible by 4, or equal to 1 for no augmentation "\
           "(default: %(default)s)")
  parser.add_argument("--threads", type=int, default=5,
      help="number of threads for dataset buffering (default: %(default)s)")
  parser.add_argument("--prefetch_batches", type=int, default=100,
      help="number of batches to prefetch (default: %(default)s)")
  parser.add_argument("--log_interval", type=int, default=100,
      help="number of steps between logging during training (default: %(default)s)")

  args = parser.parse_args(args)

  # load model
  model = load_model(args.model_path, compile=False)

  # eval!
  evaluate(args.patches_path, args.patch_size, args.batch_size, model, args.model_name,
      args.prob_threshold, args.marginalize, args.threads, args.prefetch_batches,
      args.log_interval)


if __name__ == "__main__":
  main()

