from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.sql import SparkSession
import logging
import argparse
from pyspark import SparkContext, SparkConf

from pyspark.ml.image import ImageSchema
from tensorflowonspark import TFCluster
from datetime import datetime

from dist.utils import toNpArray, genBinaryFileRDD, image_decoder, get_hdfs, read_images

import dist.mitosis_dist as mitosis_dist


def main(args=None):

  spark = SparkSession \
    .builder \
    .appName("mitosis_spark") \
    .getOrCreate()
  sc = spark.sparkContext

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1
  num_ps = 1
  logging.info("============= Num of executors: {0}".format(num_executors))

  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--appName", default="mitosis_spark", help="application name")
  parser.add_argument("--hdfs_host", help="HDFS host", type=str, default="default")
  parser.add_argument("--hdfs_port", help="HDFS port", type=int, default=8020)
  parser.add_argument("--mitosis_img_dir", help="path to the mitosis image files")
  parser.add_argument("--mitosis_img_csv", help="csv file that contain all the mitosis image files")
  parser.add_argument("--normal_img_dir", required=True, help="path to the normal image files")
  parser.add_argument("--normal_img_csv", help="csv file that contain all the normal image files")

  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=32)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
  parser.add_argument("--export_dir", help="HDFS path to export saved_model",
                      default="mnist_export")
  parser.add_argument("--format", help="example format: (csv|pickle|tfr)",
                      choices=["csv", "pickle", "tfr"], default="csv")
  parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                      default="mnist_model")
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int,
                      default=num_executors)
  parser.add_argument("--output", help="HDFS path to save test/inference output",
                      default="predictions")
  parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
  parser.add_argument("--steps", help="maximum number of steps", type=int, default=99)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--mode", help="train|inference", default="train")
  parser.add_argument("--rdma", help="use rdma connection", default=False)
  args = parser.parse_args(args)

  if args.mitosis_img_dir is None and args.mitosis_img_csv is None:
    parser.error("at least one of --mitosis_img_dir and --mitosis_img_csv required")

  if args.normal_img_dir is None and args.normal_img_csv is None:
    parser.error("at least one of --normal_img_dir and --normal_img_csv required")

  if args.mitosis_img_csv is None:
    fs = get_hdfs(args.hdfs_host, args.hdfs_port)
    mitosis_img_pathes = fs.ls(args.mitosis_img_dir)
    mitosis_label_img_pathes = [(1, path) for path in mitosis_img_pathes]
    #mitosis_train_rdd = sc.parallelize(mitosis_img_pathes).map(lambda path : (1, path))
  else:
    mitosis_train_rdd = sc.read.textFile(args.mitosis_img_csv).map(lambda path : (1, path))

  if args.normal_img_csv is None:
    fs = get_hdfs(args.hdfs_host, args.hdfs_port)
    normal_img_pathes = fs.ls(args.normal_img_dir)
    normal_label_img_pathes = [(0, path) for path in normal_img_pathes]
    #normal_train_rdd = sc.parallelize(normal_img_pathes).map(lambda path : (0, path))
  else:
    normal_train_rdd = sc.read.textFile(args.normal_img_csv).map(lambda path : (0, path))

  # get the train data set with mitosis and normal images. In the output RDD,
  # each entry will be (label, img_arr)
  training_data = []
  training_data.extend(mitosis_label_img_pathes)
  training_data.extend(normal_label_img_pathes)
  print("+++++++++++ Training data size: {}".format(len(training_data)))
  data_RDD = sc.parallelize(training_data) \
    .repartition(int(len(training_data)/128/2000)) \
    .mapPartitions(lambda iter : read_images(get_hdfs(args.hdfs_host, args.hdfs_port), iter))

  cluster = TFCluster.run(sc, mitosis_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                          TFCluster.InputMode.SPARK, log_dir=args.model)

  if args.mode == "train":
    cluster.train(data_RDD, args.epochs)
  else:
    labelRDD = cluster.inference(data_RDD)
    labelRDD.saveAsTextFile(args.output)

  cluster.shutdown(grace_secs=30)

  print("{0} ===== Stop".format(datetime.now().isoformat()))


if __name__ == "__main__":
  main()


