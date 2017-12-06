"""Detection - mitosis detection"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from breastcancer.evaluation import list_files, get_file_id, get_locations_from_csv
from breastcancer.evaluation import GROUND_TRUTH_FILE_ID_RE


def ijv_2_arr(ijv, h, w):
  """Convert an IJV array with (row,col,val) rows to a 2D array in
  which [row,col] = val.

  Args:
    ijv: A NumPy array with 3 columns corresponding to (row,col,val)
      values.
    h: The number of rows in the resulting array.
    w: The number of columns in the resulting array.

  Returns:
    A 2D NumPy array of shape (h, w) in which [row,col] = val.
  """
  arr = np.zeros((h, w), dtype=np.float32)
  arr[ijv[:,0].astype(np.int), ijv[:,1].astype(np.int)] = ijv[:,2]
  return arr


def arr_2_ijv(arr):
  """Convert a 2D array in which [row,col] = val to an IJV array with
  (row,col,val) rows.

  Args:
    arr: A 2D NumPy array of shape (h, w) in which [row,col] = val.

  Returns:
    An IJV NumPy array with 3 columns corresponding to (row,col,val)
    values.
  """
  r, c = np.nonzero(arr >= 0)
  ijv = np.vstack((r, c, arr[r,c])).T  # row, col, prob
  return ijv


def disk_kernel(radius):
  """Create a disk kernel.

  Args:
    radius: Integer value for the radius of the disk kernel.

  Returns:
    A disk kernel in which values within a distance of
    radius^2 from the center of the kernel are equal to
    one.
  """
  d = radius
  coords = np.arange(-d, d + 1)
  x, y = np.meshgrid(coords, coords, indexing="ij")
  kernel = np.array((x**2 + y**2) <= d**2, dtype=np.int)
  return kernel


def conv_smooth(probs, radius):
  """Smooth a probability map by convolving it with a disk kernel.

  This convolves the probs map with a disk kernel to produce average
  values for each pixel location.

  Args:
    probs: A 2D float32 TensorFlow tensor containing probability values.
    radius: Integer value for the radius of the disk kernel.

  Returns:
    A 2D float32 TensorFlow tensor of the same shape as `probs`
    containing smoothed probability values.
  """
  d = radius

  # create kernel
  kernel = disk_kernel(d)
  kernel = kernel / np.count_nonzero(kernel)
  kernel = kernel.reshape((*(kernel.shape), 1, 1))  # shape (h,w,c,f)

  # conv graph
  probs = tf.expand_dims(probs, 0)  # shape (n,h,w)
  probs = tf.expand_dims(probs, -1)  # shape (n,h,w,c)
  probs_padded = tf.pad(probs, paddings=[[0,0], [d, d], [d, d], [0,0]], mode="SYMMETRIC")
  probs_conv_padded = tf.nn.conv2d(probs_padded, kernel, strides=(1,1,1,1), padding="SAME")
  probs_conv = probs_conv_padded[0, d:-d, d:-d, 0]

  return probs_conv


def smooth_prediction_results(pred_dir, img_dir, radius, hasHeader):
  """Smooth the probability maps by convolving them with a disk kernel.

  This will read csv prediction files, and output smoothed csv
  prediction files.

  Args:
    pred_dir: Directory containing the prediction results
    img_dir: Directory containing the images
    radius: Integer value for the radius of the disk kernel.
    hasHeader: Boolean value to indicate if the csv file has the header
  """
  pred_files = list_files(pred_dir, "*.csv")
  pred_files = get_file_id(pred_files, GROUND_TRUTH_FILE_ID_RE)

  img_files = list_files(img_dir, "*.tif")
  img_files = get_file_id(img_files, GROUND_TRUTH_FILE_ID_RE)

  # create conv smoothing graph
  probs_tf = tf.placeholder(shape=[None, None], dtype=tf.float32)
  probs_smooth_tf = conv_smooth(probs_tf, radius)

  with tf.Session() as sess:
    for k, pred_file in pred_files.items():
      # TODO: use new `csv_2_arr` function
      preds = np.array(get_locations_from_csv(pred_file, hasHeader=hasHeader, hasProb=True))

      # convert ijv predictions to prob maps
      img_file = img_files[k]
      h, w = Image.open(img_file).size
      probs = ijv_2_arr(preds, h, w)

      # smooth the probability maps
      probs_smooth = sess.run(probs_smooth_tf, feed_dict={probs_tf: probs})

      # convert smoothed prob map back to ijv predictions
      preds_smooth = arr_2_ijv(probs_smooth)

      # save the prediction results
      smooth_dir = os.path.dirname(pred_dir + "/") + "_smoothed"
      smooth_file_name = pred_file.replace(pred_dir, smooth_dir)
      df = pd.DataFrame(preds_smooth, columns=['row', 'col', 'prob'])
      dir = os.path.dirname(smooth_file_name)
      os.makedirs(dir, exist_ok=True)
      df.to_csv(smooth_file_name, index=False)


def identify_mitoses(probs, radius, prob_thresh):
  """Identify coordinates of mitosis centers.

  This identifies mitoses through an iterative procedure as follows:

    1. Initialize a candidate list to the empty set.
    2. While there exist probabilities exceeding `prob_thresh`:
      2a. Identify the pixel `p_max` with the largest probability value
        in the probability map, and add the corresponding (row,col,prob)
        tuple to the candidate list.
      2b. Assign the value 0 to all pixels `p` for which
        `|p - p_max| < 2radius`.

  Reference:
    1. Cireşan DC, Giusti A, Gambardella LM, Schmidhuber J. Mitosis
       Detection in Breast Cancer Histology Images with Deep Neural
       Networks. Berlin, Heidelberg: Springer, Berlin, Heidelberg;
       2013. pp. 411–8. Available from:
       http://people.idsia.ch/~juergen/miccai2013.pdf

  Args:
    probs: A 2D NumPy float32 array containing probability values.
    radius: Integer value for the radius of the disk kernel.
    prob_thresh: A floating value representing the lower bound on the
      probability values.

  Returns:
    A list of (row, col, prob) tuples corresponding to the centers of
    the detected mitoses.
  """
  d = radius
  candidates = []  # (row, col, prob)

  while np.count_nonzero(probs > prob_thresh) > 0:
    # get max prob + coords, and add to candidates list
    p_max = np.max(probs)
    r, c = np.unravel_index(np.argmax(probs), probs.shape)
    assert probs[r,c] == p_max
    candidates.append((r,c,p_max))

    # suppress local non-maxima
    h, w = probs.shape
    y, x = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing="ij")
    mask = np.sqrt((y-r)**2 + (x-c)**2) >= 2*d
    probs = probs * mask

  return candidates


def detect_prediction_results(pred_dir, img_dir, radius, prob_thresh, hasHeader):
  """Detect mitoses from probability maps through an iterative
  procedure.

  This will read csv prediction files, and output csv prediction files
  containing coordinates of the predicted mitoses centers.

  Args:
    pred_dir: Directory containing the prediction results
    img_dir: Directory containing the images
    radius: Integer value for the radius of the disk kernel.
    prob_thresh: A floating value representing the lower bound on
      the probability values.
    hasHeader: Boolean value to indicate if the csv file has the header
  """
  pred_files = list_files(pred_dir, "*.csv")
  pred_files = get_file_id(pred_files, GROUND_TRUTH_FILE_ID_RE)

  img_files = list_files(img_dir, "*.tif")
  img_files = get_file_id(img_files, GROUND_TRUTH_FILE_ID_RE)

  for k, pred_file in pred_files.items():
    # TODO: use new `csv_2_arr` function
    preds = np.array(get_locations_from_csv(pred_file, hasHeader=hasHeader, hasProb=True))

    # convert ijv predictions to prob maps
    img_file = img_files[k]
    h, w = Image.open(img_file).size
    probs = ijv_2_arr(preds, h, w)

    # detect the centers of the mitoses
    preds_detected = identify_mitoses(probs, radius, prob_thresh)

    # save the prediction results
    detected_dir = os.path.dirname(pred_dir + "/") + "_detected"
    detected_file_name = pred_file.replace(pred_dir, detected_dir)
    df = pd.DataFrame(preds_detected, columns=['row', 'col', 'prob'])
    dir = os.path.dirname(detected_file_name)
    os.makedirs(dir, exist_ok=True)
    df.to_csv(detected_file_name, index=False)

