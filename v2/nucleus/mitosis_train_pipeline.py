import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
import tensorflow as tf

from deephistopath.detection import tuple_2_csv, dbscan_clustering, \
                                    cluster_prediction_result
from deephistopath.evaluation import add_ground_truth_mark_help
from deephistopath.evaluation import get_locations_from_csv, \
                                     evaluate_global_f1, \
                                     get_data_from_csv
from deephistopath.visualization import Shape
from preprocess_mitoses import extract_patch
from preprocess_mitoses import save_patch
from train_mitoses import normalize, create_augmented_batch
from v2.nucleus import nucleus_mitosis
import train_mitoses


class MitosisTrainConfig(object):
    # step_1: Reorganize the data structure for Mask_RCNN
    mitosis_input_dir = 'data/mitoses/mitoses_train_image_data/'
    mitosis_reorganized_dir = 'data/mitoses/mitoses_train_image_data_reorganized/'

    # step_2: Crop big images into small ones.
    inference_input_dir = 'data/mitoses/train/mitoses_train_image_data_cropped'
    crop_image_size = 512
    crop_image_overlap = 16

    # step_3: Run the nucleus segmentation inference
    weights = 'experiments/models/mask_rcnn_nucleus_0380.h5'
    maskrcnn_inference_result_dir = 'data/mitoses/train/mitoses_train_maskrcnn_inference_result'

    # step_4: Combine small inference images and csvs to big ones
    maskrcnn_inference_combined_result = \
        'data/mitoses/train/mitoses_train_maskrcnn_inference_result_combined/'

    # The dir of cluster result is hard coded inside function
    # `cluster_prediction_result()`.
    maskrcnn_inference_combined_clusterd_result = \
      "data/mitoses/train/mitoses_train_maskrcnn_inference_result_combined_clustered/"
    # Indicate whether the output csv file has the prediction probability
    # column.
    hasProb = True

    # step_5: Visualize the ground truth masks
    ground_truth_dir = 'data/mitoses/mitoses_train_ground_truth'

    # step_6: Evaluate the nucleus inference result

    # step_7: Label the detected nucleus as mitosis or not based on the ground
    # truth

    # step_8: Generate the mitosis and normal tiles based on the label csv

    # the mitosis patch should be bigger than mitosis tile as the patch will be
    # augmented (e.g. rotation) and crop into the mitosis tile.
    mitosis_patch_size = 72
    augmentation_number = 8
    mitosis_tile_size = 64
    mitosis_classification_prefetch = 512  # parameter for tf.dataset.prefetch
    mitosis_classification_num_parallel_calls = 8  # parameter for tf.dataset.map
    classification_model_dir = "classification_model_dir"

    # step_9: Split the datasets into train and validation folders
    extracted_normal_mitosis_patch_dir = 'data/mitoses/train/train_extracted_normal_mitosis_patches'
    val_cases = [11, 12, 15, 21, 22, 26, 29, 35, 40, 43, 51, 54, 60, 65, 68]


    # step_10: Train the classification model
    exp_parent_path = 'experiments/mitoses/mask_rcnn_retrained_classification'
    model = 'resnet_custom'
    train_batch_size = 128
    val_batch_size = 128
    clf_epochs = 100
    finetune_epochs = 10
    clf_lr = 0.0007710907499102973
    finetune_lr = 2.480308399022205e-07
    finetune_momentum = 0.8821868749724882
    finetune_layers = -1
    l2 = 2.5148154537240952e-05
    reg_biases = True
    augment = True
    oversample = True
    num_gpus = 1
    threads = 5
    prefetch_batches = 100
    log_interval = 1000


def reorganize_mitosis_images(input_dir, output_dir):
    input_files = [str(f) for f in Path(input_dir).glob('**/*.tif')]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file in input_files:
        file_basename = os.path.basename(file).split('.')[0]
        parent_dir = os.path.dirname(file)
        dir_basename = os.path.basename(parent_dir)
        new_dir = os.path.join(output_dir, "{}-{}".format(dir_basename, file_basename), 'images')
        os.makedirs(new_dir, exist_ok=True)
        copyfile(file, os.path.join(new_dir, "{}-{}.tif".format(dir_basename, file_basename)))
        print(file_basename, dir_basename)


def crop_image(input_imgs, output_dir, size=128, overlap=0):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for input_img in input_imgs:
        basename = os.path.basename(input_img).split('.')[0]
        img = cv2.imread(input_img)

        h, w, c = img.shape
        y = 0
        while y < h:
            y1 = y
            y2 = y1 + size
            if y2 > h:
                y2 = h
                y1 = h - size
            x = 0
            while x < w:
                x1 = x
                x2 = x1 + size
                if x2 > w:
                    x2 = w
                    x1 = w - size
                crop_img = img[y1: y2, x1: x2]
                result_basename = os.path.join("{}_{}_{}".format(basename, y1, x1))
                result_dir = os.path.join(output_dir, result_basename, 'images')
                os.makedirs(result_dir, exist_ok=True)
                output_file = os.path.join(result_dir, result_basename + '.png')
                print(output_file)
                cv2.imwrite(output_file, crop_img)
                x = x + size - overlap
            y = y + size - overlap


def combine_images(input_dir, output_dir, size, clean_output_dir=False):
    if clean_output_dir:
        shutil.rmtree(output_dir)
    input_files = [str(f) for f in Path(input_dir).glob('**/**/*.png')]
    input_basenames = [os.path.basename(input_file).split('.')[0].split("_")
                       for input_file in input_files]
    combined_imgs = {}
    combined_file_size = {}

    for input_basename in input_basenames:
        basename, y, x = input_basename
        y = int(y)
        x = int(x)
        if not basename in combined_file_size:
            combined_file_size[basename] = (0,0)

        combined_file_size[basename] = \
            (max(combined_file_size[basename][0], y+size),
             max(combined_file_size[basename][1], x+size))

    for basename in combined_file_size:
        h, w = combined_file_size[basename]
        combined_imgs[basename] = np.zeros([h, w, 3], dtype=np.uint8)

    for input_file in input_files:
        img = cv2.imread(input_file)
        h, w, c = img.shape
        basename, y, x = os.path.basename(input_file).split('.')[0].split("_")
        y = int(y)
        x = int(x)
        combined_imgs[basename][y:y+h, x:x+w, :] = img
        combined_file_size[basename] = \
            (max(combined_file_size[basename][0], y*size+h),
             max(combined_file_size[basename][1], x*size+w))

    os.makedirs(output_dir, exist_ok=True)
    for combined_img in combined_imgs:
        cv2.imwrite(os.path.join(output_dir, combined_img) + '.png',
                    combined_imgs[combined_img][
                    0:combined_file_size[basename][0],
                    0:combined_file_size[basename][1], :])


def combine_csvs(input_dir, output_dir, hasHeader=True, hasProb=True,
                 clean_output_dir=False):
    if clean_output_dir:
        shutil.rmtree(output_dir)
    input_files = [str(f) for f in Path(input_dir).glob('**/**/*.csv')]
    combine_csvs = {}
    for input_file in input_files:
        points = get_locations_from_csv(input_file, hasHeader=hasHeader,
                                     hasProb=hasProb)
        basename, y_offset, x_offset = \
            os.path.basename(input_file).split('.')[0].split("_")
        if not basename in combine_csvs:
            combine_csvs[basename] = []
        y_offset = int(y_offset)
        x_offset = int(x_offset)
        for i in range(len(points)):
            points[i] = \
                (points[i][0] + y_offset, points[i][1] + x_offset, points[i][2])
        combine_csvs[basename].extend(points)

    os.makedirs(output_dir, exist_ok=True)
    for combined_csv in combine_csvs:
        tuple_2_csv(combine_csvs[combined_csv],
                    os.path.join(output_dir, combined_csv) + '.csv',
                    columns=['Y', 'X', 'prob'])


def add_groundtruth_mark(im_dir, ground_truth_dir, hasHeader=False,
                         shape=Shape.CROSS, mark_color=(0, 255, 127, 200),
                         hasProb=False):
    im_files = [str(f) for f in Path(im_dir).glob('*.png')]
    for im_file in im_files:
        im_file_basename = os.path.basename(im_file).split('.')[0]
        ground_truth_file_path = os.path.join(ground_truth_dir,
                                              *im_file_basename.split("-"))
        ground_truth_file_path = "{}.csv".format(ground_truth_file_path)
        if not os.path.exists(ground_truth_file_path):
            print("{} doestn't exist".format(ground_truth_file_path))
            continue
        add_ground_truth_mark_help(im_file, ground_truth_file_path,
                                   hasHeader=hasHeader, shape=shape)


def add_mark(img_file, csv_file, hasHeader=False, shape=Shape.CROSS,
             mark_color=(0, 255, 127, 200), hasProb=False):
    add_ground_truth_mark_help(img_file, csv_file, hasHeader=hasHeader,
                               shape=shape, mark_color=mark_color,
                               hasProb=hasProb)


def is_inside(x1, y1, x2, y2, radius):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist <= radius


def check_nucleus_inference(inference_dir, ground_truth_dir):
    ground_truth_csvs = [str(f) for f in Path(ground_truth_dir).glob('*/*.csv')]
    matched_count = 0
    total_count = 0
    for ground_truth_csv in ground_truth_csvs:
        ground_truth_dir, base = os.path.split(ground_truth_csv)
        sub_dir = os.path.split(ground_truth_dir)[1]
        inference_csv = os.path.join(inference_dir, "{}-{}".format(sub_dir, base))
        ground_truth_locations = get_locations_from_csv(
            ground_truth_csv, hasHeader=False, hasProb=False)
        inference_locations = get_locations_from_csv(
            inference_csv, hasHeader=True, hasProb=False)
        for (x1, y1) in ground_truth_locations:
            total_count = total_count + 1
            for (x2, y2) in inference_locations:
                if is_inside(x2, y2, x1, y1, 32):
                    matched_count = matched_count + 1
                    break
    print("There are {} ground truth points, found {} of them.".format(
        total_count, matched_count))


def label_detected_nucleus(nucleus_dir, ground_truth_dir):
    ground_truth_csvs = [str(f) for f in Path(ground_truth_dir).glob('*/*.csv')]
    label_output = []
    for ground_truth_csv in ground_truth_csvs:
        ground_truth_dir, base = os.path.split(ground_truth_csv)
        sub_dir = os.path.split(ground_truth_dir)[1]
        inference_csv = os.path.join(nucleus_dir, "{}-{}".format(sub_dir, base))
        label_csv = os.path.join(nucleus_dir,
          "label-{}-{}".format(sub_dir, base))
        ground_truth_locations = get_locations_from_csv(
            ground_truth_csv, hasHeader=False, hasProb=False)
        inference_locations = get_locations_from_csv(
            inference_csv, hasHeader=True, hasProb=True)
        label_output.clear()
        for (y1, x1, prob) in inference_locations:
          inside = False
          for (y2, x2) in ground_truth_locations:
            if is_inside(x1, y1, x2, y2, 32):
              inside = True
              label_output.append((y1, x1, prob, True))
              break
          if not inside:
            label_output.append((y1, x1, prob, False))
        print(len(label_output), len(inference_locations))
        assert len(label_output) == len(inference_locations)
        tuple_2_csv(label_output, label_csv, ['Y', 'X', 'prob', 'is_mitosis'])


def gen_mitosis_normal_tiles(img_dir, location_csv_dir, output_patch_basedir,
                             tile_size=64):
    normal_count = 0
    mitosis_count = 0
    location_csv_files = [str(f) for f in Path(location_csv_dir).glob('label-*.csv')]
    if len(location_csv_files) == 0:
        raise ValueError(
            "Please check the input dir for the location csv files.")

    for location_csv_file in location_csv_files:
        print("Processing {} ......".format(location_csv_file))
        points = get_data_from_csv(location_csv_file, hasHeader=True)
        if len(points) == 0:
          raise ValueError("No data found, please check the input file {}"
            .format(location_csv_file))
        if len(points[0]) != 4:
          raise ValueError("Please check the data schema which should have 4 "
                           "columns")
        # Get the image file name.
        subfolder = os.path.basename(location_csv_file) \
            .replace('label-', '') \
            .replace('-', '/') \
            .replace('.csv', '')
        case, region = subfolder.split('/')
        img_file = os.path.join(img_dir, "{}.tif".format(subfolder))
        print("Processing {} ......".format(img_file))
        img = cv2.imread(img_file)
        img = np.asarray(img)[:, :, ::-1]

        for (row, col, prob, is_mitosis) in points:
            output_patch_dir = os.path.join(output_patch_basedir,
                                            'mitosis' if is_mitosis else 'normal')
            if not os.path.exists(output_patch_dir):
              os.makedirs(output_patch_dir, exist_ok=True)

            patch = extract_patch(img, row, col, tile_size)
            save_patch(patch, path=output_patch_dir, lab=0, case=case,
                       region=region, row=row, col=col, rotation=0, row_shift=0,
                       col_shift=0, suffix=0, ext="png")
            if is_mitosis:
              mitosis_count += 1
            else:
              normal_count += 1
        print("Extract {} mitoses and {} normal titles".format(mitosis_count,
                                                               normal_count))


def get_image_tf(filename):
  """Get image from filename.

  Args:
    filename: String filename of an image.

  Returns:
    TensorFlow tensor containing the decoded and resized image with
    type float32 and values in [0, 1).
  """
  image_string = tf.read_file(filename)
  # shape (h,w,c), uint8 in [0, 255]:
  image = tf.image.decode_png(image_string, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image


def get_location_from_file_name(filename):
    basename = os.path.basename(str(filename))
    filename_comps = basename.split('_')
    row = int(filename_comps[3])
    col = int(filename_comps[4])
    return row, col


def run_mitosis_classification(model,
                               sess,
                               batch_size,
                               input_dir_path,
                               output_dir_path,
                               augmentation_number,
                               mitosis_tile_size=64,
                               num_parallel_calls=1,
                               prefetch=32,
                               prob_thres=0.5,
                               eps=64, min_samples=1,
                               isWeightedAvg=False):

    input_file_paths = [str(f) for f in Path(input_dir_path).glob('*.png')]
    input_files = np.asarray(input_file_paths, dtype=np.str)

    input_file_dataset = tf.data.Dataset.from_tensor_slices(input_files)
    img_dataset = input_file_dataset.map(lambda file: get_image_tf(file),
                                         num_parallel_calls=1)

    if augmentation_number == 1:
      img_dataset = img_dataset\
        .map(lambda img: normalize(img, "resnet_custom"),
             num_parallel_calls=num_parallel_calls)\
        .batch(batch_size)\
        .prefetch(prefetch)
      # Make sure all the files in the dataset are feeded into inference
      float_steps = len(input_file_paths) / batch_size
      int_steps = len(input_file_paths) // batch_size
      steps = math.ceil(float_steps) if float_steps > int_steps else int_steps
    else:
      img_dataset = img_dataset \
        .map(lambda img: create_augmented_batch(img, augmentation_number,
                         mitosis_tile_size),
             num_parallel_calls=num_parallel_calls) \
        .map(lambda img: normalize(img, "resnet_custom"),
             num_parallel_calls=num_parallel_calls) \
        .prefetch(prefetch)
      steps = len(input_file_paths)

    img_iterator = img_dataset.make_one_shot_iterator()
    next_batch = img_iterator.get_next()

    while True:
        try:
            pred_np = model.predict(next_batch, steps=steps)
            print("Prediction result shape: ", pred_np.shape)
        except tf.errors.OutOfRangeError:
            print("Please check the steps parameter. steps = {}, "
                  "batch_size = {}, input_tile_size = {}, "
                  "augmentation_number = {}"
                  .format(steps, batch_size, input_files.shape,
                          augmentation_number))
            break

    prob_result = \
      np.average(pred_np.reshape(-1, augmentation_number), axis=1)

    print("Finish the inference on {} with {} input tiles"
          .format(input_dir_path, prob_result.shape))

    assert prob_result.shape[0] == input_files.shape[0]
    mitosis_probs = prob_result[prob_result > prob_thres]
    input_files = input_files.reshape(-1, 1)
    mitosis_patch_files = input_files[prob_result > prob_thres]
    inference_result = []
    for i in range(mitosis_patch_files.size):
        row, col = get_location_from_file_name(mitosis_patch_files[i])
        prob = mitosis_probs[i]
        inference_result.append((row, col, prob))

    if len(inference_result) > 0:
        clustered_pred_locations = dbscan_clustering(
            inference_result, eps=eps, min_samples=min_samples,
            isWeightedAvg=isWeightedAvg)
        tuple_2_csv(
            inference_result,
            os.path.join(output_dir_path, 'mitosis_locations.csv'))
        tuple_2_csv(
            clustered_pred_locations,
            os.path.join(output_dir_path, 'clustered_mitosis_locations.csv'))
    else:
        print("Do not have mitosis in {}".format(input_dir_path))


def load_model(model_file):
    # create session
    config = tf.ConfigProto(
        allow_soft_placement=True)  # , log_device_placement=True)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # load the model and add the sigmoid layer
    base_model = tf.keras.models.load_model(model_file, compile=False)

    # specify the name of the added activation layer to avoid the name
    # conflict in ResNet
    probs = tf.keras.layers.Activation('sigmoid', name="sigmoid")(
        base_model.output)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=probs)

    return model, sess


def run_mitosis_classification_in_batch(batch_size,
                                        input_dir_basepath,
                                        output_dir_basepath,
                                        model_file,
                                        augmentation_number,
                                        mitosis_tile_size=64,
                                        num_parallel_calls=1,
                                        prefectch= 32,
                                        prob_thres=0.5,
                                        eps=64, min_samples=1,
                                        isWeightedAvg=True):
    re = '[0-9]'*2 + '/' + '[0-9]'*2
    input_patch_dirs = [str(f) for f in Path(input_dir_basepath).glob(re)]

    model, sess = load_model(model_file)
    for input_patch_dir in input_patch_dirs:
        print("Run the inference on {} ......".format(input_patch_dir))
        input_patch_dir = Path(input_patch_dir)
        subfolder = os.path.join(input_patch_dir.parent.name,
                                 input_patch_dir.name)
        reference_output_dir = os.path.join(output_dir_basepath, subfolder)
        run_mitosis_classification(
            model, sess, batch_size, input_patch_dir, reference_output_dir,
            augmentation_number, mitosis_tile_size, num_parallel_calls,
            prefectch, prob_thres, eps, min_samples, isWeightedAvg)


def split_train_val_datasets(input_dir, val_cases):
  mitosis_dir = os.path.join(input_dir, 'mitosis')
  normal_dir = os.path.join(input_dir, 'normal')

  train_dir = os.path.join(input_dir, 'train')
  train_mitosis_dir = os.path.join(train_dir, 'mitosis')
  train_normal_dir = os.path.join(train_dir, 'normal')
  os.makedirs(train_mitosis_dir, exist_ok=True)
  os.makedirs(train_normal_dir, exist_ok=True)

  val_dir = os.path.join(input_dir, 'val')
  val_mitosis_dir = os.path.join(val_dir, 'mitosis')
  val_normal_dir = os.path.join(val_dir, 'normal')
  os.makedirs(val_mitosis_dir, exist_ok=True)
  os.makedirs(val_normal_dir, exist_ok=True)

  mitosis_files = [str(f) for f in Path(mitosis_dir).glob('*.png')]
  normal_files = [str(f) for f in Path(normal_dir).glob('*.png')]

  for mitosis_file in mitosis_files:
    is_val = False
    for val_case in val_cases:
      if mitosis_file.find("0_{:0>2d}".format(val_case)) != -1:
        shutil.copy(mitosis_file, val_mitosis_dir)
        is_val = True
        break
    if not is_val:
      shutil.copy(mitosis_file, train_mitosis_dir)

  for normal_file in normal_files:
    is_val = False
    for val_case in val_cases:
      if normal_file.find("0_{:0>2d}".format(val_case)) != -1:
        shutil.copy(normal_file, val_normal_dir)
        is_val = True
        break
    if not is_val:
      shutil.copy(normal_file, train_normal_dir)


def main(args):
    config = MitosisTrainConfig()
    if args.reorganize_folder_structure:
        print("1. Reorganize the data structure for Mask_RCNN")
        reorganize_mitosis_images(
            config.mitosis_input_dir, config.mitosis_reorganized_dir)

    if args.split_big_images_to_small_ones:
        print("2. Crop big images into small ones")
        mitosis_files = \
            [str(f) for f in Path(config.mitosis_reorganized_dir)
                                 .glob('**/**/*.tif')]
        crop_image(mitosis_files, config.inference_input_dir,
                   size=config.crop_image_size,
                   overlap=config.crop_image_overlap)

    if args.run_nucleus_detection:
        print("3. Run the nucleus segmentation inference")
        nucleus_mitosis.inference(config.inference_input_dir, config.weights,
                                  config.maskrcnn_inference_result_dir)
        print("Inference result dir: ", config.maskrcnn_inference_result_dir)

    if args.combine_small_inference_results:
        print("4. Combine small inference images and csvs to big ones")
        combine_images(config.maskrcnn_inference_result_dir,
                       config.maskrcnn_inference_combined_result,
                       size=config.crop_image_size)
        combine_csvs(config.maskrcnn_inference_result_dir,
                     config.maskrcnn_inference_combined_result,
                     hasProb=config.hasProb, clean_output_dir=False)

    if args.cluster_nucleus_detection_results:
        print("4.1 Cluster the nucleus detection results")
        cluster_prediction_result(config.maskrcnn_inference_combined_result,
                                  eps=32, min_samples=1, hasHeader=True,
                                  isWeightedAvg=False, prob_threshold=0.8)
        # for file_name in ["11-01", "11-02", "11-03", "12-01", "12-02", "12-03"]:
        #   add_mark(os.path.join(config.maskrcnn_inference_combined_result,
        #                         "{}.png".format(file_name)),
        #            os.path.join(config.maskrcnn_inference_combined_clusterd_result,
        #                         "{}.csv".format(file_name)),
        #            hasHeader=True, shape=Shape.SQUARE,
        #            mark_color=(255, 100, 100, 200))

    if args.visualize_the_ground_truth:
        print("5. Visualize the ground truth masks")
        add_groundtruth_mark(config.maskrcnn_inference_combined_result,
                             config.ground_truth_dir, hasHeader=False,
                             shape=Shape.CIRCLE)
        # add_mark('/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_combine_test/01-01.png',
        #          '/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_combine_test/01-01.csv',
        #          hasHeader=True, shape=Shape.CIRCLE, mark_color=(255,0,0,50))

    if args.evaluate_nucleus_inference:
        print("6. Evaluate the nucleus inference result")
        check_nucleus_inference(config.maskrcnn_inference_combined_result,
                                 config.ground_truth_dir)

    if args.label_detected_nucleus:
        print("7. Label the detected nucleus as mitosis or not based on the "
              "ground truth")
        label_detected_nucleus(config.maskrcnn_inference_combined_result,
                               config.ground_truth_dir)

    if args.gen_mitosis_normal_tile:
        print("8. Generate the mitosis and normal tiles based on the label csv")
        gen_mitosis_normal_tiles(config.mitosis_input_dir,
                                  config.maskrcnn_inference_combined_result,
                                  config.extracted_normal_mitosis_patch_dir,
                                  tile_size=config.mitosis_tile_size)
    if args.split_train_val_datasets:
        print("9. Split the datasets into train and validation folders")
        split_train_val_datasets(config.extracted_normal_mitosis_patch_dir,
                                 config.val_cases)

    if args.train_model:
        print("10. Train the classification model")
        train_args = []
        train_args.append("--patches_path={}".format(config.extracted_normal_mitosis_patch_dir))
        train_args.append("--exp_parent_path={}".format(config.exp_parent_path))
        train_args.append("--model={}".format(config.model))
        train_args.append("--model_weights={}".format(config.mitosis_classification_model_file))
        train_args.append("--train_batch_size={}".format(config.train_batch_size))
        train_args.append("--val_batch_size={}".format(config.val_batch_size))
        train_args.append("--clf_epochs={}".format(config.clf_epochs))
        train_args.append("--finetune_epochs={}".format(config.finetune_epochs))
        train_args.append("--clf_lr={}".format(config.clf_lr))
        train_args.append("--finetune_lr={}".format(config.finetune_lr))
        train_args.append("--finetune_momentum={}".format(config.finetune_momentum))
        train_args.append("--finetune_layers={}".format(config.finetune_layers))
        train_args.append("--l2={}".format(config.l2))
        if config.reg_biases:
          train_args.append("--reg_biases")
        if config.augment:
          train_args.append("--augment")
        if config.oversample:
          train_args.append("--oversample")
        train_args.append("--num_gpus={}".format(config.num_gpus))
        train_args.append("--threads={}".format(config.threads))
        train_args.append("--prefetch_batches={}".format(config.prefetch_batches))
        train_args.append("--log_interval={}".format(config.log_interval))

        try:
          train_mitoses.main(train_args)
        except tf.errors.InvalidArgumentError as e:  # if values become nan or inf
          print(e)
          print("Experiment failed!")


if __name__ == '__main__':
    import argparse
    start_time = time.time()
    logging.info("Start the job ")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Deep-Histopath: Nucleus Detection")
    parser.add_argument("--reorganize_folder_structure", required=False, action="store_true",
                        default=False, help="Reorganize the data folder as "
                                            "required by Mask R-CNN")
    parser.add_argument("--split_big_images_to_small_ones", required=False, action="store_true",
                        default=False, help="Split big images to small ones")
    parser.add_argument("--run_nucleus_detection", required=False, action="store_true",
                        default=False, help="Detect the nuclei from the input "
                                            "images")
    parser.add_argument("--combine_small_inference_results", required=False, action="store_true",
                        default=False, help="Combine the inference results (csv "
                                            "and image) as the whole result for "
                                            "the big image")
    parser.add_argument("--cluster_nucleus_detection_results", required=False, action="store_true",
                        default=False, help="The detected nucleus might be much "
                                            "smaller than the predefiend "
                                            "mitosis tile size (64), so the "
                                            "detected nuclei could be clustered "
                                            "and the center point can be used "
                                            "to extract the potential mitosis.")
    parser.add_argument("--visualize_the_ground_truth", required=False, action="store_true",
                        default=False, help="Visualize the ground truth data on "
                                            "the big image")
    parser.add_argument("--evaluate_nucleus_inference", required=False, action="store_true",
                        default=False, help="Evaluate the nucleus inference "
                                            "result to see how many detected "
                                            "nuclei are overlapped with the "
                                            "ground truth mitoses")
    parser.add_argument("--label_detected_nucleus", required=False,
                         action="store_true", default=False,
                         help="Add a label for the detected nuclei based on "
                              "the ground truth data: True means the detected "
                              "nucleus is a motosis; False means it is not a "
                              "motosis")

    parser.add_argument("--gen_mitosis_normal_tile", required=False, default=False,
                        action="store_true", help="Extract the patches from the "
                                                  "train images based on the "
                                                  "input coordinates and split "
                                                  "the data into mitosis and "
                                                  "normal forlder")

    parser.add_argument("--split_train_val_datasets", required=False,
                        default=False, action="store_true",
                        help="Split the tiles into train and validation folers")

    parser.add_argument("--run_mitosis_classification", required=False, action="store_true",
                        default=False, help="Run the mitosis detection model to"
                                            "classify if the input image is a "
                                            "mitosis or not")
    parser.add_argument("--compute_f1", required=False, default=False, action="store_true",
                        help="Compute F1 score")
    args = parser.parse_args()
    main(args)
    end_time = time.time() - start_time
    print("This job took: {} seconds".format(end_time))
    logging.info("This job took: {} seconds".format(end_time))



