"""Hyperparameter tuning - mitosis detection"""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
import keras.backend as K

import train_mitoses


def main(args=None):
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--patches_path", required=True,
      help="path to the generated image patches containing `train` & `val` folders ")
  parser.add_argument("--exp_parent_path",
      default=os.path.join("experiments", "mitoses", "hyp"),
      help="parent path in which to store experiment folders (default: %(default)s)")
  parser.add_argument("--models", nargs='*', default=["vgg", "resnet"],
      help="list of names of models to use, where the names can be selected from ['logreg', "\
           "'vgg', 'vgg19', 'resnet'] (default: %(default)s)")
  parser.add_argument("--train_batch_sizes", nargs='*', type=int, default=[32],
      help="list of training batch sizes (default: %(default)s)")
  parser.add_argument("--val_batch_size", type=int, default=32,
      help="validation batch size for all experiments (default: %(default)s)")
  parser.add_argument("--clf_epochs", type=int, default=5,
      help="number of epochs for which to train the new classifier layers (default: %(default)s)")
  parser.add_argument("--finetune_epochs", type=int, default=5,
      help="number of epochs for which to fine-tune the unfrozen layers (default: %(default)s)")
  parser.add_argument("--clf_lr_range", nargs=2, type=float, default=(1e-5, 1e-2),
      help="half-open interval for the learning rate for training the new classifier layers "\
           "(default: %(default)s)")
  parser.add_argument("--finetune_lr_range", nargs=2, type=float, default=(1e-7, 1e-2),
      help="half-open interval for the learning rate for fine-tuning the unfrozen layers "\
           "(default: %(default)s)")
  parser.add_argument("--finetune_momentum_range", nargs=2, type=float, default=(0.85, 0.95),
      help="half-open interval for the momentum rate for fine-tuning the unfrozen layers "\
           "(default: %(default)s)")
  parser.add_argument("--finetune_layers", nargs='*', type=int, default=[0, -1],
      help="list of the number of layers at the end of the pretrained portion of the model to "\
           "fine-tune (note: the new classifier layers will still be trained during fine-tuning "\
           "as well) (default: %(default)s)")
  parser.add_argument("--l2_range", nargs=2, type=float, default=[0, 1e-2],
      help="half-closed interval for the amount of l2 weight regularization (default: %(default)s)")
  augment_parser = parser.add_mutually_exclusive_group(required=False)
  augment_parser.add_argument("--augment", dest="augment", action="store_true",
      help="apply random augmentation to the training images (default: True)")
  augment_parser.add_argument("--no_augment", dest="augment", action="store_false",
      help="do not apply random augmentation to the training images (default: False)")
  parser.set_defaults(augment=True)
  parser.add_argument("--marginalize", default=False, action="store_true",
      help="use noise marginalization when evaluating the validation set. if this is set, then "\
           "the validation batch_size must be divisible by 4, or equal to 1 for no augmentation "\
           "(default: %(default)s)")
  parser.add_argument("--threads", type=int, default=5,
      help="number of threads for dataset parallel processing; note: this will cause "\
           "non-reproducibility for values > 1 (default: %(default)s)")
  parser.add_argument("--prefetch_batches", type=int, default=100,
      help="number of batches to prefetch (default: %(default)s)")
  parser.add_argument("--log_interval", type=int, default=100,
      help="number of steps between logging during training (default: %(default)s)")
  parser.add_argument("--num_experiments", type=int, default=100,
      help="number of experiments to run (default: %(default)s)")

  args = parser.parse_args(args)

  # hyperparameter search
  for i in range(args.num_experiments):
    # NOTE: as a quick POC, we will use the command-line interface of the training script
    # TODO: extract experiment setup code in the training script main function into a class so that
    # we can reuse it from here
    train_args = []
    train_args.append(f"--patches_path={args.patches_path}")

    train_args.append(f"--exp_parent_path={args.exp_parent_path}")

    model = random.choice(args.models)
    train_args.append(f"--model={model}")

    train_batch_size = random.choice(args.train_batch_sizes)
    train_args.append(f"--train_batch_size={train_batch_size}")

    train_args.append(f"--val_batch_size={args.val_batch_size}")

    train_args.append(f"--clf_epochs={args.clf_epochs}")

    train_args.append(f"--finetune_epochs={args.finetune_epochs}")

    clf_lr_lb, clf_lr_ub = args.clf_lr_range
    clf_lr = np.random.uniform(clf_lr_lb, clf_lr_ub)
    train_args.append(f"--clf_lr={clf_lr}")

    finetune_lr_lb, finetune_lr_ub = args.finetune_lr_range
    finetune_lr = np.random.uniform(finetune_lr_lb, finetune_lr_ub)
    train_args.append(f"--finetune_lr={finetune_lr}")

    finetune_momentum_lb, finetune_momentum_ub = args.finetune_momentum_range
    finetune_momentum = np.random.uniform(finetune_momentum_lb, finetune_momentum_ub)
    train_args.append(f"--finetune_momentum={finetune_momentum}")

    finetune_layers = random.choice(args.finetune_layers)
    train_args.append(f"--finetune_layers={finetune_layers}")

    l2_lb, l2_ub = args.l2_range
    l2 = np.random.uniform(l2_lb, l2_ub)
    train_args.append(f"--l2={l2}")

    if args.augment:
      train_args.append("--augment")
    else:
      train_args.append("--no_augment")

    if args.marginalize:
      train_args.append("--marginalize")

    train_args.append(f"--threads={args.threads}")

    train_args.append(f"--prefetch_batches={args.prefetch_batches}")

    train_args.append(f"--log_interval={args.log_interval}")

    # train!
    try:
      train_mitoses.main(train_args)
    except tf.errors.InvalidArgumentError:  # if values become nan or inf
      print("Experiment failed!")

    # it is necessary to completely reset everything in between experiments
    K.clear_session()


if __name__ == "__main__":
  main()

