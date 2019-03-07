from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from pyspark.sql.types import BinaryType, StringType, StructField, StructType
from PIL import Image
from io import BytesIO
import logging
import pyarrow as pa
import socket


def toNpArray(row):
  """
  Converts an image row in DataFrame to Numpy array
  :param row: A row that contains the image to be converted.
  :return:
  """
  image = row[0]
  height = image.height
  width = image.width
  nChannels = image.nChannels

  return np.ndarray(
    shape=(height, width, nChannels),
    dtype=np.uint8,
    buffer=image.data,
    strides=(width * nChannels, nChannels, 1))

def image_decoder(rawbytes):
  """
  Decode raw bytes to image array
  :param rawbytes:
  :return: a numpy array with uint8
  """
  img = Image.open(BytesIO(rawbytes))
  array = np.asarray(img, dtype=np.uint8)
  return array


def genBinaryFileRDD(sc, path, numPartitions=None):
  """
    Read files from a directory to a RDD.
    :param sc: SparkContext.
    :param path: str, path to files.
    :param numPartition: int, number or partitions to use for reading files.
    :return: RDD with a pair of key and value: (filePath: str, fileData: BinaryType)
    """
  numPartitions = numPartitions or sc.defaultParallelism
  rdd = sc.binaryFiles(
    path, minPartitions=numPartitions).repartition(numPartitions)
  #rdd = rdd.map(lambda x: (x[0], bytearray(x[1])))
  return rdd

def get_hdfs(host, port):
  """
  Connect to HadoopFileSystem
  :param host: HDFS namenode host
  :param port: HDFS namenode port, which can be retrieved by `hdfs getconf -nnRpcAddresses`
  :return: HadoopFileSystem
  """
  fs = pa.hdfs.connect(host, port)
  return fs

def read_image(fs, img_path, mode="rb"):
  """
  Read image file from HDFS
  :param fs: HadoopFileSystem
  :param img_path: image file path
  :param mode: The mode.  If given, this argument must be "r"
  :return:
  """
  f = fs.open(img_path, mode)
  pil_img = Image.open(f)
  img_array = np.asarray(pil_img, dtype=np.uint8)
  f.close()
  return img_array

def read_images(fs, img_path_batch, mode="rb"):
  """
  Read images from HDFS in batch
  :param fs: HadoopFileSystem
  :param img_path_batch: a batch of image pathes
  :param mode: The mode
  :return: a list of numpy array
  """
  result = []
  logging.info("Start to read images at {}".format(socket.gethostname()))
  for (label, img_path) in img_path_batch:
    img = read_image(fs, img_path, mode)
    result.append((label, img))
  logging.info("Finish the reading of {} images on {}".format(
    len(result), socket.gethostname()))
  return result


