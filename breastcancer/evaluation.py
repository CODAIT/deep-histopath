"""
Evaluate the prediction results
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
from PIL import Image
from breastcancer.visualization import Shape, add_mark

GROUND_TRUTH_FILE_ID_RE = "\d+/\d+"

def compute_f1(predict, ground_truth, threshold=30):
  """ Compute the F1 score, the calculation equation can be found at
  http://amida13.isi.uu.nl/?q=node/4.

  The implementation idea is to compute TP, FP, and FN first, and then
  get F1 score. TP: its Eucledian distance to a ground truth location
  is less than threshold; FP: not within the threshold distance of a
  ground truth location; FN: all ground truth locations that do not
  have a detection within the threshold.

  Args:
    predict: prediction result, a list of point locations, e.g.
      [(r0, c0), (r1, c1), (r2, c2), ...].
    ground_truth: groud truth data, a list of point locations, e.g.
      [(r0, c0), (r1, c1), (r2, c2), ...].
    threshold: Eucledian distance to see if the predict and ground truth
      points are at the same circle.

  Return:
    F1 score.
  """
  # initialize the values.
  FP = predict.copy()
  TP = []
  FN = []

  # check which points in the ground truth are located with each other
  # less than `threshold`.
  for gt_r, gt_c in ground_truth:
    for r, c in ground_truth:
      dist = np.sqrt((gt_r - r) ** 2 + (gt_c - c) ** 2)
      if dist > 0 and dist < threshold * 2:
        print(f"Point ({r} , {c}) has multiple points in the circle")

  for gt_r, gt_c in ground_truth:
    # if several points fall within a single ground truth location,
    # they will be counted as one true positive. Here we use a list
    # to collect this kind of points for each ground truth point.
    tp = []
    for pred_r, pred_c in predict:
      dist = np.sqrt((gt_r - pred_r)**2 + (gt_c - pred_c)**2)
      if dist <= threshold:
        tp.append((pred_r, pred_c))
        # remove the TP point from the FP list
        if (pred_r, pred_c) in FP:
          FP.remove((pred_r, pred_c))
    if len(tp) == 0:
      # if no points fall within this ground truth point, the ground
      # truth point will be treated as FN.
      FN.append((gt_r, gt_c))
    else:
      TP.append(tp)

  precision = len(TP) / (len(TP) + len(FP))
  recall = len(TP) / (len(TP) + len(FN))
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
  return f1

def list_files(dir, file_suffix):
  """ recursively list all the files that have the same input file
    suffix under the input directory.

  Args
    dir: file directory.
    file_suffix: file suffix.

  Return:
    a list of file path.
  """
  dir = Path(dir)
  files = [str(x) for x in dir.rglob(file_suffix)]
  return files

def get_file_id(files, file_id_re):
  """ get the file id using the file id regular expression
  Args:
    files: input file paths
    file_id_re: regular expression string used to detect the file ID
      from the file path

  Return:
    a dictionary of file id and its full file path
  """
  id_files = {re.findall(file_id_re, x)[0]: x for x in files}
  return id_files


def get_locations_from_csv(file, hasHeader=False):
  """ get the point locations from CSV file.

  Args:
    file: csv file, of which the first and second columns store the
      point coordinates.
    hasHeader: bool value to tell if the input csv file has a header or
      not.

  Return:
    a list of point locations, e.g. [(r0, c0), (r1, c1), ......].
  """
  if hasHeader:
    data = pd.read_csv(file)
  else:
    data = pd.read_csv(file, header=None)

  locations = [(x[0], x[1]) for x in data.values.tolist()]
  return locations

def evaluate(pred_dir, ground_true_dir, threshold=30):
  """ Evaluate the prediction result based on the ground truth data
    using F1 score.

  Args:
    pred_dir: directory for the prediction csv files.
    ground_true_dir: directory for the ground truth csv files.
    threshold: Eucledian distance to see if the predict and ground
      truth points are at the same circle.

  Return:
     a tuple of (a F1_score list for each input file; a list of files
      that are detected by model, but not in the ground truth data; a
      list of files that are not detected by model, but in the ground
      truth data).
  """
  pred_files = list_files(pred_dir, "*.csv")
  pred_files = get_file_id(pred_files, GROUND_TRUTH_FILE_ID_RE)
  ground_true_files = list_files(ground_true_dir, "*.csv")
  ground_true_files = get_file_id(ground_true_files, GROUND_TRUTH_FILE_ID_RE)

  f1_list = []
  over_detected = []
  non_detected = []
  for k, pred_file in pred_files.items():
    if k in ground_true_files:
      ground_true = ground_true_files[k]
      pred_locations = get_locations_from_csv(pred_file, hasHeader=True)
      ground_true_location = get_locations_from_csv(ground_true, hasHeader=False)
      f1 = compute_f1(pred_locations, ground_true_location, threshold)
      f1_list.append([f"{k}-{len(ground_true_location)}-{len(pred_locations)}", f1])
    else:
      # record the image IDs that the model detected but not in the ground truth
      over_detected.append(k)

  # record the image IDs that are not detected by the model
  for k in ground_true_files.keys():
    if k not in pred_files:
      non_detected.append(k)

  return (f1_list, over_detected, non_detected)

def add_ground_truth_mark_help(im_path, ground_truth_file_path, hasHeader=False, shape=Shape.CROSS):
  """ add the mark for each point in the ground truth data.

  Args:
    im_path: image file path to add the marks.
    ground_truth_file_path: file path for the ground truth csv file.
    hasHeader: bool value to tell if the ground truth csv file has a header.
    shape: options for mark shape. It could be Shape.CROSS, Shape.SQUARE,
      or Shape.CIRCLE.
  """
  im = Image.open(im_path)
  locations = get_locations_from_csv(ground_truth_file_path, hasHeader=hasHeader)
  add_mark(im, locations, shape)
  im.save(im_path)

def add_ground_truth_mark(sparkContext, partition_num, im_dir, im_suffix, ground_truth_dir,
                          ground_truth_file_suffix, shape=Shape.CROSS, hasHeader=False):
  """ Add the ground truth data as marks into the images in parallel.

    Args:
      sparkContext: Spark context.
      partition_num: number of partitions for the input images
      im_dir: input image directory.
      im_suffix: input image suffix.
      ground_truth_dir: ground truth file directory
      ground_truth_file_suffix: ground truth file suffix
      shape: options for mark shape. It could be Shape.CROSS, Shape.SQUARE,
        or Shape.CIRCLE.
      hasHeader: bool value to tell if the ground truth csv file has a
        header.
    """

  # assume that the image file path and ground truth file path are like
  # "/File/.../Path/ID1/ID2/FileName"
  input_imgs = list_files(im_dir, im_suffix)
  id_imgs = get_file_id(input_imgs, GROUND_TRUTH_FILE_ID_RE)

  ground_truth_files = list_files(ground_truth_dir, ground_truth_file_suffix)
  id_ground_truth_files = get_file_id(ground_truth_files, GROUND_TRUTH_FILE_ID_RE)

  # match the image data with the ground truth csv file
  img_ground_truth = []
  for id_ground_truth in id_ground_truth_files.keys():
    if id_ground_truth in id_imgs:
      img_ground_truth.append((id_imgs[id_ground_truth], id_ground_truth_files[id_ground_truth]))

  # parallel the input images, and add the marks to each image
  rdd = sparkContext.parallelize(img_ground_truth, partition_num)
  rdd.foreach(lambda t: add_ground_truth_mark_help(t[0], t[1], hasHeader=hasHeader, shape=shape))


def test_compute_f1():
  threshold = 30

  # test case 1: the prediction result is the same with the ground truth
  predict1 = [(10, 10), (20, 30), (30, 60), (100, 53), (32, 26), (120, 66)]
  ground_true1 = [(10, 10), (20, 30), (30, 60), (100, 53), (32, 26), (120, 66)]
  f1 = compute_f1(predict1, ground_true1, threshold)
  assert f1 == 1

  # test case 2: the prediction result is totally different from the ground truth
  predict2 = [(10, 10), (2, 3), (3, 6), (10, 5), (3, 2), (12, 6)]
  ground_true2 = [(100, 100), (200, 300), (300, 600), (1000, 530), (320, 260), (1200, 660)]
  f1 = compute_f1(predict2, ground_true2, threshold)
  assert f1 == 0

  # test case 3: the prediction result partially matches the ground truth
  predict2 = [(2, 3), (50, 180), (66, 20), (80, 70), (100, 200), (300, 400)]
  ground_true2 = [(10, 10), (40, 50), (50, 200), (60, 110), (70, 10), (80, 80)]
  f1 = compute_f1(predict2, ground_true2, threshold)
  assert f1 == 2/3


def test_add_ground_truth_mark():
  from pyspark.sql import SparkSession
  # Create new SparkSession
  spark = SparkSession.builder.appName("Add marks").getOrCreate()
  sparkContext = spark.sparkContext
  im_dir = "result/mitoses/mitoses_train_image_result"
  partition_num = 4
  im_suffix = "*mark.tif"
  ground_truth_dir = "data/mitoses/mitoses_ground_truth"
  ground_truth_file_suffix = "*.csv"
  hasHeader = False

  add_ground_truth_mark(sparkContext, partition_num, im_dir, im_suffix, ground_truth_dir,
                        ground_truth_file_suffix, Shape.CIRCLE, hasHeader)


def test_img_quality():
  # this test shows the image quality difference between different image
  # formats (e.g., tif, png, jpeg) and different libraries (e.g., openslide,
  # PIL, Tensorflow) with different reading parameters. The experiment
  # results indicate that tif and png have the highest quality; JPEG changes
  # a lot of pixel values when it compresses the images, but has the smallest
  # file size.

  import os
  import openslide
  import tensorflow as tf
  from breastcancer.preprocessing import create_tile_generator, get_20x_zoom_level
  import pandas as pd

  # all the output images from these reader functions are in the type of
  # np.int instead of np.uint8, which avoids the invalid value when
  # computing the difference between two images

  def read_img_openslide(img_file):
    slide = openslide.open_slide(img_file)
    ROI_size = max(slide.dimensions)
    generator = create_tile_generator(slide, ROI_size, 0)
    zoom_level = get_20x_zoom_level(slide, generator)
    cols, rows = generator.level_tiles[zoom_level]
    ROI_indices = [(zoom_level, col, row) for col in range(cols) for row in range(rows)]
    zl, col, row = ROI_indices[0]
    img_openslide = np.asarray(generator.get_tile(zl, (col, row)), dtype=np.int)
    return img_openslide

  def read_img_PIL(img_file):
    img_PIL = Image.open(img_file)
    img_PIL = np.array(img_PIL, dtype=np.int)
    return img_PIL

  def read_jpeg_tf_fast(jpg_file):
    image_string = tf.read_file(jpg_file)
    img_tensor_fast = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_FAST')

    with tf.Session() as sess:
      img_tf_fast = sess.run(img_tensor_fast).astype(np.int)
    return img_tf_fast

  def read_jpeg_tf_accurate(jpg_file):
    image_string = tf.read_file(jpg_file)
    img_tensor_accurate = tf.image.decode_jpeg(image_string, channels=3,
                                               dct_method='INTEGER_ACCURATE')
    with tf.Session() as sess:
      img_tf_accurate = sess.run(img_tensor_accurate).astype(np.int)
    return img_tf_accurate

  def read_png_tf(png_file):
    image_string = tf.read_file(png_file)
    img_tensor_png = tf.image.decode_png(image_string, channels=3, dtype=tf.uint8)
    with tf.Session() as sess:
      img_tf_png = sess.run(img_tensor_png).astype(np.int)
    return img_tf_png

  def compute_diff(img1, img2, threshold=0):
    diff = np.sum(np.abs(img1 - img2) > threshold)/img1.size
    return diff

  # generate an image array
  img_orig = np.random.randint(low=0, high=256, size=(2000, 2000, 3), dtype=np.uint8)

  # save the image array as the tif file
  tif_file = "test.tif"
  Image.fromarray(img_orig).save(tif_file, subsampling=0, quality=100)

  img_tif = Image.open(tif_file)
  # save the image array from the tif file into: 1) low-quality jpeg:
  # using the default configuration of PIL; 2) high-quality jpeg:
  # adding the customized configuration for PIL to generate a
  # high-quality JPG; 3) png
  jpg_lq_file = "test_lq.jpg"
  jpg_hq_file = "test_hq.jpg"
  png_file = "test.png"
  img_tif.save(jpg_lq_file)
  img_tif.save(jpg_hq_file, subsampling=0, quality=100)
  img_tif.save(png_file)
  img_tif.close()

  # start to test the reading result from OpenSlide, PIL, and Tensorflow

  # convert the data type of original image from uint8 to int
  img_orig = img_orig.astype(np.int)

  # read images using openslide
  img_openslide_jpg_lq = read_img_openslide(jpg_lq_file)
  img_openslide_jpg_hq = read_img_openslide(jpg_hq_file)
  img_openslide_tif = read_img_openslide(tif_file)
  img_openslide_png = read_img_openslide(png_file)

  # read images using PIL
  img_PIL_jpg_lq = read_img_PIL(jpg_lq_file)
  img_PIL_jpg_hq = read_img_PIL(jpg_hq_file)
  img_PIL_tif = read_img_PIL(tif_file)
  img_PIL_png = read_img_PIL(png_file)

  # read images using Tensorflow
  img_tf_jpg_lq_fast = read_jpeg_tf_fast(jpg_lq_file)
  img_tf_jpg_lq_accurate = read_jpeg_tf_accurate(jpg_lq_file)
  img_tf_jpg_hq_fast = read_jpeg_tf_fast(jpg_hq_file)
  img_tf_jpg_hq_accurate = read_jpeg_tf_accurate(jpg_hq_file)
  img_tf_png = read_png_tf(png_file)

  # evaluate the difference between different images using different
  # libraries
  img_name_list = ["orig_img", "openslide_tif", "PIL_tif", "openslide_png", "PIL_png", "tf_png",
                   "openslide_jpg_lq", "openslide_jpg_hq", "PIL_jpg_lq", "PIL_jpg_hq",
                   "tf_jpg_lq_fast", "tf_jpg_lq_accurate", "tf_jpg_hq_fast", "tf_jpg_hq_accurate"]

  img_dict = {"orig_img": img_orig,
               "openslide_tif": img_openslide_tif,
               "PIL_tif": img_PIL_tif,
               "openslide_png": img_openslide_png,
               "PIL_png": img_PIL_png,
               "tf_png": img_tf_png,
               "openslide_jpg_lq": img_openslide_jpg_lq,
               "openslide_jpg_hq": img_openslide_jpg_hq,
               "PIL_jpg_lq": img_PIL_jpg_lq,
               "PIL_jpg_hq": img_PIL_jpg_hq,
               "tf_jpg_lq_fast": img_tf_jpg_lq_fast,
               "tf_jpg_lq_accurate": img_tf_jpg_lq_accurate,
               "tf_jpg_hq_fast": img_tf_jpg_hq_fast,
               "tf_jpg_hq_accurate": img_tf_jpg_hq_accurate}

  num_img = len(img_name_list)
  diff = np.empty([num_img, num_img], dtype=np.float32)

  # compute the difference ratio between different images
  threshold = 0 # threshold to judge whether two pixels are different
  for row in range(num_img):
    for col in range(num_img):
      diff[row][col] = compute_diff(img_dict[img_name_list[row]],
                                    img_dict[img_name_list[col]],
                                    threshold=threshold)

  # organize the comparison result into data frame, and print it out as
  # table
  df = pd.DataFrame(diff, img_name_list, img_name_list)
  with pd.option_context("display.max_rows", num_img, "display.max_columns", num_img,
                         'expand_frame_repr', False):
    print(df)

  # delete the temporary file
  os.remove(tif_file)
  os.remove(jpg_hq_file)
  os.remove(jpg_lq_file)
  os.remove(png_file)
