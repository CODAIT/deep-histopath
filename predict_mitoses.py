"""
This script predict the number of mitoses for the input slide images
"""
import os
import argparse
import shutil
from pyspark.sql import SparkSession
from breastcancer.inference import predict_mitoses

def flat_result_2_row(predictions):
  """ flat the mitosis prediction result into rows

  Args:
    predictions: a tuple of (slide_id, ROI, mitosis_num,
      mitosis_location_scores), where mitosis_location_scores is a list
      of tuples (r, c, score)
  Return:
    a list of tuples(slide_id, ROI, mitosis_num, r, c, score)
  """

  result = []
  if predictions:
    for pred in predictions:
      slide_id, ROI, mitosis_num, mitosis_location_scores = pred
      for r, c, score in mitosis_location_scores:
        result.append((slide_id, ROI, mitosis_num, r, c, score))
  return result


def main(args=None):
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--appName", default="Breast Cancer -- Predict",
                      help="application name (default: %(default)s)")
  parser.add_argument("--slide_path", default=os.path.join("data", "training_image_data"),
                      help="path to the mitosis data for prediction (default: %(default)s)")
  parser.add_argument("--model_path", default=os.path.join("model",
                                                           "0.95114_acc_0.58515_loss_530_epoch_model.hdf5"),
                      help="path to the model file (default: %(default)s)")
  parser.add_argument("--model_name", default="vgg",
                      help="input model type, e.g. vgg, resnet (default: %(default)s)")
  parser.add_argument("--file_suffix", default="*.svs",
                      help="file suffix for the input data set, e.g. *.svs (default: %(default)s)")
  parser.add_argument("--node_number", type=int, default=2,
                      help="number of available computing node in the spark cluster "\
                           "(default: %(default)s)")
  parser.add_argument("--gpu_per_node", type=int, default=4,
                      help="number of GPUs on each computing node (default: %(default)s)")
  parser.add_argument("--cpu_per_node", type=int, default=4,
                      help="number of CPUs on each computing node (default: %(default)s)")
  parser.add_argument("--ROI_size", type=int, default=6000,
                      help="size of ROI (default: %(default)s)")
  parser.add_argument("--ROI_overlap", type=int, default=0,
                      help="overlap between ROIs (default: %(default)s)")
  parser.add_argument("--ROI_channel", type=int, default=3,
                      help="number of ROI channel (default: %(default)s)")
  parser.add_argument("--skipROI", default=False, dest='skipROI', action='store_true',
                      help="skip the ROI layer (default: %(default)s)")
  parser.add_argument("--tile_size", type=int, default=64,
                      help="size of tile (default: %(default)s)")
  parser.add_argument("--tile_overlap", type=int, default=0,
                      help="overlap between tiles (default: %(default)s)")
  parser.add_argument("--tile_channel", type=int, default=3,
                      help="channel of tile (default: %(default)s)")
  parser.add_argument("--mitosis_threshold", type=float, default=0.5,
                      help="the threshold for the identification of mitosis (default: %(default)s)")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="batch size for the mitosis prediction (default: %(default)s)")
  parser.add_argument("--marginalize", default=False, action="store_true",
                      help="use noise marginalization when evaluating the validation set. if this "\
                           "is set, then the `batch_size` must be divisible by 4, or equal to 1 "\
                           "for a special debugging case of no augmentation (default: %(default)s)")
  parser.add_argument("--onGPU", dest='isGPU', action='store_true',
                      help="run the script on GPU (default: False)")
  parser.add_argument("--onCPU", dest='isGPU', action='store_false',
                      help="run the script on CPU (default: True)")
  parser.set_defaults(isGPU=False)
  parser.add_argument("--save_mitosis_locations", default=False, dest="save_mitosis_locations",
                      action='store_true',
                      help="save the locations of the detected mitoses to csv "\
                           "(default: %(default)s)")
  parser.add_argument("--save_mask", default=False, dest="save_mask", action='store_true',
                      help="save the locations of the detected mitoses as a mask image "\
                           "(default: %(default)s)")
  parser.add_argument("--pred_save_path", required=True,
                      help="file path to save the prediction results")
  parser.add_argument("--debug", default=False, dest='isDebug', action='store_true',
                      help="print the debug information (default: %(default)s)")

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
  sparkContext.addPyFile("preprocess_mitoses.py")
  sparkContext.addPyFile("resnet50.py")
  predict_result_rdd = predict_mitoses(sparkContext=sparkContext, model_path=args.model_path,
                                       model_name=args.model_name,
                                       input_dir=args.slide_path, file_suffix=args.file_suffix,
                                       partition_num=args.partition_num,
                                       ROI_size=args.ROI_size, ROI_overlap=args.ROI_overlap,
                                       ROI_channel=args.ROI_channel, skipROI=args.skipROI,
                                       tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                                       tile_channel=args.tile_channel,
                                       threshold=args.mitosis_threshold, isGPU=args.isGPU,
                                       batch_size=args.batch_size,
                                       marginalization=args.marginalize,
                                       save_mitosis_locations=args.save_mitosis_locations,
                                       save_mask=args.save_mask, isDebug=args.isDebug)

  pred_rows = predict_result_rdd.flatMap(lambda t: flat_result_2_row(t)).cache()

  df = spark.createDataFrame(pred_rows, ['slide_id', 'ROI_id', 'mitosis_num_per_ROI', 'row_num',
                                         'col_num', 'score'])
  df.toPandas().to_csv(args.pred_save_path, header=True)
  df.show()

if __name__ == "__main__":
  main()
