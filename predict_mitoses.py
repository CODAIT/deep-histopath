"""
This script predict the number of mitoses for the input slide images
"""
import os
import argparse
import shutil
from pyspark.sql import SparkSession
from breastcancer.inference import predict_mitoses

if __name__ == "__main__":
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--appName", default="Breast Cancer -- Predict", help="application name")
  parser.add_argument("--slide_path", default=os.path.join("data", "training_image_data"),
                      help="path to the mitosis data for prediction")
  parser.add_argument("--model_path", default=os.path.join("model", "0.95114_acc_0.58515_loss_530_epoch_model.hdf5"),
                      help="path to the model file")
  parser.add_argument("--model_name", default="vgg", help="input model type, e.g. vgg, resnet")
  parser.add_argument("--file_suffix", default=".svs", help="file suffix for the input data set, e.g. *.svs")
  parser.add_argument("--node_number", type=int, default=2,
                      help="number of available computing node in the spark cluster")
  parser.add_argument("--gpu_per_node", type=int, default=4,
                      help="number of GPUs on each computing node")
  parser.add_argument("--cpu_per_node", type=int, default=4,
                      help="number of CPUs on each computing node")
  parser.add_argument("--ROI_size", type=int, default=6000, help="size of ROI")
  parser.add_argument("--ROI_overlap", type=int, default=0, help="overlap between ROIs")
  parser.add_argument("--ROI_channel", type=int, default=3, help="number of ROI channel")
  parser.add_argument("--tile_size", type=int, default=64, help="size of tile")
  parser.add_argument("--tile_overlap", type=int, default=0, help="overlap between tiles")
  parser.add_argument("--tile_channel", type=int, default=3, help="channel of tile")
  parser.add_argument("--mitosis_threshold", type=float, default=0.5,
                      help="the threshold for the identification of mitosis")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size for the mitosis prediction")
  parser.add_argument("--onGPU", dest='isGPU', action='store_true',
                      help="run the script on GPU")
  parser.add_argument("--onCPU", dest='isGPU', action='store_false',
                      help="run the script on CPU")
  parser.set_defaults(isGPU=False)
  parser.add_argument("--debug", dest='isDebug', action='store_true', default=False,
                      help="print the debug information")

  args = parser.parse_args()
  if args.isGPU:
    args.partition_num = args.gpu_per_node * args.node_number
  else:
    args.partition_num = args.cpu_per_node * args.node_number

  # Create new SparkSession
  spark = (SparkSession.builder
                       .appName(args.appName)
                       .getOrCreate())
  sparkContext = spark.sparkContext

  # Ship a fresh copy of the `breastcancer` package to the Spark workers.
  # Note: The zip must include the `breastcancer` directory itself,
  # as well as all files within it for `addPyFile` to work correctly.
  # This is equivalent to `zip -r breastcancer.zip breastcancer`.
  dirname = "breastcancer"
  zipname = dirname + ".zip"
  shutil.make_archive(dirname, 'zip', dirname + "/..", dirname)
  sparkContext.addPyFile(zipname)
  sparkContext.addPyFile("train_mitoses.py")

  predict_result_rdd = predict_mitoses(sparkContext=sparkContext, model_path=args.model_path, model_name = args.model_name,
                                       input_dir=args.slide_path, file_suffix=args.file_suffix, partition_num=args.partition_num,
                                       ROI_size=args.ROI_size, ROI_overlap=args.ROI_overlap, ROI_channel=args.ROI_channel,
                                       tile_size=args.tile_size, tile_overlap=args.tile_overlap, tile_channel=args.tile_channel,
                                       threshold=args.mitosis_threshold, isGPU=args.isGPU, batch_size=args.batch_size, isDebug=args.isDebug)

  print(predict_result_rdd.collect())

