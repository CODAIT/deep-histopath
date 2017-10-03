"""
This script predict the number of mitoses for the input slide images
"""
import re
import socket
from collections import Counter
import numpy as np
from skimage.util.shape import view_as_windows
from time import gmtime, strftime

from pathlib import Path
import openslide

import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from breastcancer.preprocessing import create_tile_generator, get_20x_zoom_level
from train_mitoses import normalize


def check_subsetting(ROI, ROI_size, tiles, tile_size, tile_overlap, channel=3):
  """ check if the generation of tiles is right by re-combine the tiles

  Args:
    ROI: The original ROI image (h, w, c).
    ROI_size: The size of ROI.
    tiles: The tiles (n, h, w, c).
    tile_size: The size of tile.
    tile_overlap: The overlap between tiles.
    channel: the number of channel; ROI and tiles should have the same number of channel.
  """
  try:
    stride = tile_size - tile_overlap

    # compute the number of rows and columns when splitting ROI into tiles.
    col_num = row_num = (ROI_size - tile_size) // stride + 1
    height = width = stride * (col_num - 1) + tile_size
    tile_recombination = np.zeros((0, width, channel), dtype=np.uint8)
    for row in range(row_num):
      # for the last row, its height will be equal to the tile size; Otherwise, it will be the stride of window moving.
      h = stride if row < row_num - 1 else tile_size
      cur = np.zeros((h, 0, channel), dtype=np.uint8)
      for col in range(col_num):
        # in each row, the width of last tile will be equal to tile size;
        # for other tiles, their width will be equal to stride.
        w = stride if col < col_num - 1 else tile_size
        index = col_num * row + col
        cur = np.concatenate((cur, tiles[index][0:h, 0:w, ]), axis=1)
      tile_recombination = np.concatenate((tile_recombination, cur), axis=0)
    return np.array_equal(ROI[0:height, 0:width, ], tile_recombination)
  except:
    return False


def predict_help(model_file, model_name, index, file_partition,
                 ROI_size, ROI_overlap, ROI_channel=3,
                 tile_size=64, tile_overlap=0, tile_channel=3,
                 threshold=0.5, isGPU=True, batch_size=128, isDebug=False):
  """ Predict the number of mitoses for each input slide image.

  Args:
    model_file: The file path for the input model (.hdf5).
    index: if using GPU, it will be the assigned gpu id for this partition; if using cpu, it will the split index.
    file_partition: The partition of input files.
    ROI_size: The ROI size.
    ROI_overlap: The overlap between ROIs.
    ROI_channel: The channel of ROI.
    tile_size: The tile siz.
    tile_overlap: The overlap between tiles.
    tile_channel: The channel of tiles.
    threshold: The threshold for the output of last sigmoid layer.
    isGPU: true if running on tensorflow-GPU; false if running on tensorflow-CPU.
    batch_size: the batch_size for prediction.
    isDebug: if true, print out the debug information.

  Return:
     A list of prediction result tuple (file_path, mitoses_sum).
  """

  # configure GPU
  if isGPU:
    gpu_id = str(index)   #get_gpus(1)
    conf = tf.ConfigProto()
    conf.allow_soft_placement = True
    conf.gpu_options.visible_device_list = gpu_id
    tf_session = tf.Session(config=conf)
    K.set_session(tf_session)
    if isDebug:
      print(f"GPU_ID: {gpu_id}")

  # load the model and add the sigmoid layer
  base_model = load_model(model_file)
  probs = keras.layers.Activation('sigmoid')(base_model.output)
  model = keras.models.Model(inputs=base_model.input, outputs=probs)

  result = []

  for file_path in file_partition:
    # generate the ROI indices for each input file
    slide_id = int(re.search('(?<=TUPAC-TR-)\d+', file_path).group())
    slide = openslide.open_slide(str(file_path))
    generator = create_tile_generator(slide, ROI_size, ROI_overlap)
    zoom_level = get_20x_zoom_level(slide, generator)
    cols, rows = generator.level_tiles[zoom_level]
    ROI_indices = [(zoom_level, col, row) for col in range(cols) for row in range(rows)]

    for ROI_index in ROI_indices:
      # get the ROI
      zl, col, row = ROI_index
      ROI = np.asarray(generator.get_tile(zl, (col, row)))
      # generate the tiles for each ROI
      # TODO: the current solution needs to read the actual ROI and then generate the tiles;
      # A more efficient solution may be to generate the tile indices based on the ROI index,
      # and then directly extract the tiles from the slide by `generator.level_tiles[zoom_level]`.
      # It could avoid the reading of ROI.
      tiles = view_as_windows(ROI, (tile_size, tile_size, tile_channel), step=tile_size - tile_overlap) \
                .reshape(-1, tile_size, tile_size, tile_channel)

      # normalize the tiles. Note that if the model is vgg or resnet, the channel order is changed in the normalization
      tiles = normalize(tiles / 255, model_name)
      isMitoses = model.predict(tiles, batch_size) > threshold
      mitoses_num = np.sum(isMitoses, dtype=np.int32)
      result.append((slide_id, f"ROI_{row}_{col}", mitoses_num))
      if isDebug:
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        info = f"Slide: {slide_id}, ROI_ID: {row}_{col}, Mitoses_Num: {mitoses_num}; " \
               f"Time: {cur_time}"
        print(info)

  return result

def predict_mitoses(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel=3, tile_size=64, tile_overlap=0,
                    tile_channel=3, threshold=0.5, isGPU=True, batch_size=128, isDebug=False):
    """ Predict the number of mitoses for the input slide images. It supports both GPUs and CPUs.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs
      ROI_channel: channel of ROI.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles
      threshold: threshold for the output of last sigmoid layer.
      isGPU: true if running on tensorflow-GPU; false if running on tensorflow-CPU
      batch_size: the batch_size for prediction
      isDebug: if true, print out the debug information

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum)
    """
    if isGPU:
        predictions_rdd = predict_mitoses_gpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel, tile_size, tile_overlap,
                    tile_channel, threshold, batch_size, isDebug)
    else:
        predictions_rdd = predict_mitoses_cpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel, tile_size, tile_overlap,
                    tile_channel, threshold, batch_size, isDebug)
    return predictions_rdd


def predict_mitoses_gpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel=3, tile_size=64, tile_overlap=0,
                    tile_channel=3, threshold=0.5, batch_size=128, isDebug=False):
  """ Predict the number of mitoses for the input slide images using GPU.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs.
      ROI_channel: channel of ROI.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles.
      threshold: threshold for the output of last sigmoid layer.
      batch_size: the batch_size for prediction.
      isDebug: if true, print out the debug information.

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum).
    """

  input_dir = Path(input_dir)
  input_imgs = [str(x) for x in input_dir.rglob(file_suffix)]

  # parallel the input slide images and split it to several partitions specified by the input
  # At the same time, cache this rdd to keep the location of each partition
  slide_rdd = sparkContext.parallelize(input_imgs, partition_num).cache()

  # get the hostname for each partition
  get_hostname = lambda split_index, partitions : [(split_index, socket.gethostname())]
  split_index_2_host = slide_rdd.mapPartitionsWithIndex(get_hostname).collect()

  # TODO: there is an assumption here that the number of partitions on each node is same with the number of available GPUs
  # get the number of available GPUs on each node
  host_2_gpu_num = Counter(t[1] for t in split_index_2_host)

  # assign the gpu id to each partition
  split_index_2_gpu_id = []
  for tuple in split_index_2_host:
        host_2_gpu_num[tuple[1]] -= 1
        map = (tuple[0], host_2_gpu_num[tuple[1]])
        split_index_2_gpu_id.append(map)
  split_index_2_gpu_id = dict(split_index_2_gpu_id)

  if isDebug:
    print (host_2_gpu_num)
    print(split_index_2_host)
    print(split_index_2_gpu_id)

  # assign GPU id to each partition, and then run the predict function for each partition.
  predictions_rdd = slide_rdd.mapPartitionsWithIndex(lambda index, p: [(split_index_2_gpu_id[index], p)])\
    .map(lambda tuple : predict_help(model_file=model_path, model_name = model_name, index=tuple[0],
                                     file_partition=tuple[1], ROI_size=ROI_size, ROI_overlap=ROI_overlap, ROI_channel=ROI_channel,
                                     tile_size=tile_size, tile_overlap=tile_overlap, tile_channel=tile_channel, threshold=threshold,
                                     isGPU=True, batch_size=batch_size, isDebug=isDebug))

  return predictions_rdd

def predict_mitoses_cpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel=3, tile_size=64, tile_overlap=0,
                    tile_channel=3, threshold=0.5, batch_size=128, isDebug=False):
  """ Predict the number of mitoses for the input slide images using CPU.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs.
      ROI_channel: channel of ROI.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles.
      threshold: threshold for the output of last sigmoid layer.
      batch_size: the batch_size for prediction.
      isDebug: if true, print out the debug information.

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum).
    """

  input_dir = Path(input_dir)
  input_imgs = [str(x) for x in input_dir.rglob(file_suffix)]

  # parallel the input slide images and repartition it to be same with the number of input images
  rdd = sparkContext.parallelize(input_imgs, partition_num)

  # run the predict function for each partition.
  predictions_rdd = rdd.mapPartitionsWithIndex(lambda index, p: predict_help(model_file=model_path, model_name = model_name,
                                                                             index=index, file_partition=p,
                                                                             ROI_size=ROI_size, ROI_overlap=ROI_overlap, ROI_channel=ROI_channel,
                                                                             tile_size=tile_size, tile_overlap=tile_overlap, tile_channel=tile_channel,
                                                                             threshold=threshold, isGPU=False,
                                                                             batch_size=batch_size, isDebug=isDebug))
  return predictions_rdd

