"""Preprocessing - mitosis detection"""
import argparse
import math
import os
import shutil

from keras.models import load_model
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from train_mitoses import normalize


def create_mask(h, w, coords, size):
  """Create a binary image mask with locations of mitosis patches.

  Areas equal to zero indicate normal regions, while areas equal to one
  indicate mitosis regions.

  Args:
    h: Integer height of the mask.
    w: Integer width of the mask.
    coords: A list-like collection of (row, col) mitosis coordinates.
    size: An integer size of the square patches to place on the mask.

  Returns:
    A binary mask of the same shape as `im` indicating where the
    mitosis patches are located.
  """
  # check that row, col, and size are within the image bounds
  assert 1 < size <= min(h, w), "size must be >1 and within the bounds of the image"

  # create mitosis patch mask
  mask = np.zeros((h, w), dtype=np.bool)
  for row, col in coords:
    assert 0 <= row <= h, "row is outside of the image height"
    assert 0 <= col <= w, "col is outside of the image width"

    # (row, col) is the center, so compute upper and lower bounds of patch
    half_size = round(size / 2)
    row_lower = row - half_size
    row_upper = row + half_size
    col_lower = col - half_size
    col_upper = col + half_size

    # clip the bounds to the size of the image
    row_lower = max(0, row_lower)
    row_upper = min(row_upper, h)
    col_lower = max(0, col_lower)
    col_upper = min(col_upper, w)

    # indicate mitosis patch area on mask
    mask[row_lower:row_upper, col_lower:col_upper] = True

  return mask


def extract_patch(im, row, col, size):
  """Extract a patch centered at (row, col).

  If the (row, col) is at the edge of the image, the image will be
  reflected to yield a patch of the desired size.

  Args:
    im: An image stored as a NumPy array of shape (h, w, ...).
    row: An integer row number.
    col: An integer col number.
    size: An integer size of the square patch to extract.

  Returns:
    A NumPy array of shape (size, size, ...).
  """
  # check that row, col, and size are within the image bounds
  dims = np.ndim(im)
  assert dims >= 2, "image must be of shape (h, w, ...)"
  h, w = im.shape[0:2]
  assert 0 <= row <= h, "row is outside of the image height"
  assert 0 <= col <= w, "col is outside of the image width"
  assert 1 < size <= min(h, w), "size must be >1 and within the bounds of the image"

  # (row, col) is the center, so compute upper and lower bounds of patch
  half_size = round(size / 2)
  row_lower = row - half_size
  row_upper = row + half_size
  col_lower = col - half_size
  col_upper = col + half_size

  # clip the bounds to the size of the image and compute padding to add to patch
  row_pad_lower = abs(row_lower) if row_lower < 0 else 0
  row_pad_upper = row_upper - h if row_upper > h else 0
  col_pad_lower = abs(col_lower) if col_lower < 0 else 0
  col_pad_upper = col_upper - w if col_upper > w else 0
  row_lower = max(0, row_lower)
  row_upper = min(row_upper, h)
  col_lower = max(0, col_lower)
  col_upper = min(col_upper, w)

  # extract patch
  patch = im[row_lower:row_upper, col_lower:col_upper]

  # pad with reflection on the height and width as needed to yield a patch of the desired size
  # NOTE: all remaining dimensions (such as channels) receive 0 padding
  padding = ((row_pad_lower, row_pad_upper), (col_pad_lower, col_pad_upper)) + ((0, 0),) * (dims-2)

  # Note: the padding content starts from the second row/col of the
  # input patch instead of the first row/col
  patch_padded = np.pad(patch, padding, 'reflect')

  return patch_padded

def gen_dense_coords(h, w, size, stride):
  """Generate centered (row, col) coordinates of patches densely from an
  image with striding.

  This slides across the image from left to right, top to bottom by
  `stride` number of pixels, yielding (row, col) centered coordinates.

  Args:
    h: Integer height of the image.
    w: Integer width of the image.
    size: An integer size of the square patch to extract.
    stride: An integer number of pixels by which to shift in the
      sliding window for normal patches.

  Returns:
    Yields (row, col) integer coordinates of the center of a patch.
  """
  # check that row, col, and size are within the image bounds
  assert 1 < size <= min(h, w), "size must be > 1 and within the bounds of the image"
  assert stride > 0, "stride must be an integer > 0"

  # generate coordinates
  for row in range(0, h, stride):
    for col in range(0, w, stride):
      yield row, col  # centered coordinates for this patch


def gen_normal_coords(mask, size, stride, threshold):
  """Generate (row, col) coordinates for normal patches.

  This generates coordinates for normal patches in a sliding window
  fashion with the given stride, possibly overlapping with mitosis
  patches up to `threshold` percentage.

  Args:
    mask: A binary mask, indicating where the mitosis patches are
      located, of the same height and width as the region image.
    size: An integer size of the square patch to extract.
    stride: An integer number of pixels by which to shift in the
      sliding window for normal patches.
    threshold: A decimal inclusive upper bound on the percentage of
      allowable overlap with mitosis patches.

  Returns:
    Yields (row, col) coordinates of a normal patch.
  """
  # check that size is within the mask bounds
  assert np.ndim(mask) == 2, "mask must be of shape (h, w)"
  h, w = mask.shape
  assert 1 < size <= min(h, w), "size must be > 1 and within the bounds of the image"
  assert stride > 0, "stride must be an integer > 0"
  assert 0 <= threshold <= 1, "threshold must be a valid decimal percentage"

  for row, col in gen_dense_coords(h, w, size, stride):
    # extract patch from mask to check for overlap with mitosis patch
    mask_patch = extract_patch(mask, row, col, size)
    if np.mean(mask_patch) <= threshold:
      yield row, col


def gen_fp_coords(im, normal_coords, size, model, model_name, pred_threshold):
  """Generate (row, col) coordinates for false-positive patches.

  This generates false-positive patch coordinates by making predictions
  for each normal patch and yielding coordinates for cases in which the
  predicted value is greater than `pred_threshold`.  Note: by having the
  threshold be a parameter, the model can output the actual probability
  value or the logit value and as long as the threshold is set
  appropriately it doesn't matter (i.e., for a probability threshold of
  0.5, the corresponding logit threshold would be 0).

  Args:
    im: An image stored as a np.uint8 NumPy array of shape (h, w, c) with
      values in [0, 255].
    normal_coords: An iterable collection of (row, col) coordinates.
    size: An integer size of the square patch to extract.
    model: Keras model to use for false-positive oversampling.
    model_name: String indicating the model being used, which is used
      for determining the correct normalization.  TODO: replace this
    pred_threshold: Decimal threshold over which the patch is predicted
      as a positive case.

  Returns:
    Yields (row, col) coordinates of false-positive patches.
  """
  for row, col in normal_coords:
    patch = extract_patch(im, row, col, size)
    norm_patch = normalize(patch / 255, model_name)
    out = model.predict_on_batch(np.expand_dims(norm_patch, 0))[0]
    if out > pred_threshold:
      yield row, col


def gen_random_translation(h, w, row, col, max_shift):
  """Generate (row_shift, col_shift) random translation shifts relative
  to (row, col).

  Ensures that the shifts are within the bounds of the image.

  Args:
    h: Integer height of the image.
    w: Integer width of the image.
    row: An integer row number.
    col: An integer col number.
    max_shift: Integer upper bound on the spatial shift range for the
      random translations.

  Returns:
    New (row_shift, col_shift) integer relative translations.
  """
  # check that row, col, and size are within the image bounds
  assert 0 <= row <= h, "row is outside of the image height"
  assert 0 <= col <= w, "col is outside of the image width"
  assert max_shift >= 0, "max_shift must be >= 0"

  # NOTE: np.random.randint has exclusive upper bounds
  row_shifted = min(max(0, row + np.random.randint(-max_shift, max_shift + 1)), h)
  col_shifted = min(max(0, col + np.random.randint(-max_shift, max_shift + 1)), w)
  row_shift = row_shifted - row
  col_shift = col_shifted - col
  return row_shift, col_shift


def gen_patches(im, coords, size, rotations, translations, max_shift, p):
  """Generate patches with sampling and augmentation from coordinates.

  For every set of (row, col) coordinates in `coords`, this function
  yields centered patches sampled with probability `p`, possibly with a
  combination of some number of rotations evenly-spaced in [0, 180], and
  some number of random translations per rotation.

  NOTE: This function will internally create an uint8 version of `im`
  in order to use PIL to rotate the image.  It will yield patches
  converted back to the original type.

  Args:
    im: An image stored as a NumPy array of shape (h, w, c).
    coords: An iterable collection of (row, col) coordinates.
    size: An integer size of the square patch to extract.
    rotations: Integer number of rotation-augmented patches
      evenly-spaced in [0, 180] to extract for each location, in
      addition to a 0-degree rotation.
    translations: An integer number of random translation augmented
      patches to extract for each rotation (including the 0-degree
      rotation), in addition to a translation of 0.
    max_shift: Integer upper bound on the spatial shift range for
      the random translations.
    p: A decimal probability of sampling each patch.

  Returns:
    Yields (patch, row, col, rot, row_shift, col_shift) tuples, where
    patch is a NumPy array of shape (size, size, c), row & col are the
    original coordinates, rot is the degree of rotation, and row_shift
    & col_shift are relative translations from row, col applied after
    the rotation.
  """
  # check that size is within the image bounds
  assert np.ndim(im) == 3, "image must be of shape (h, w, c)"
  h, w, c = im.shape
  assert 1 < size <= min(h, w), "size must be > 1 and within the bounds of the image"
  assert rotations >= 0, "rotations must be >0"
  assert translations >= 0, "translations must be >0"
  assert max_shift >= 0, "max_shift must be >= 0"
  assert 0 <= p <= 1, "p must be a valid decimal probability"

  # convert to uint8 type in order to use PIL to rotate
  orig_dtype = im.dtype
  im = im.astype(np.uint8)

  # We want to extract a rotated image, but if we simply extract a patch and rotate it,
  # the corners will be empty.  Ideally, we don't want to have empty corners, or have
  # to fill those corners in with random noise, mirroring, etc.  Since we have access
  # to the full image, we can first extract a larger patch from which a regular patch
  # size can be extracted after a rotation without including any empty regions.
  # If we rotate a patch by theta degrees, then the corners of a centered square patch
  # will intersect with the sides of the rotated larger patch at the same angle theta,
  # forming a right triangle between the side of the centered patch as the hypotenuse,
  # the segment of the side of the rotated patch between the corner and the intersection
  # with the centered patch, the corner of the rotated patch, and the segment on the
  # next side, which is the complement of the first segment lengthwise. Since we know
  # the angle and the length of the centered patch, we can compute the lengths of the
  # two segments, and thus the length of the side of the outer patch.  A 45 degree
  # rotation is the worst case scenario, so we extract a bounding patch for that case.
  # Additionally, to support random translations, we add `2*max_shift` to the length,
  # and then simply adjust the center coordinates and rotate around that shifted center.
  # Instead of random rotations, we extract evenly-spaced rotations in the range [0, 180],
  # starting with 0 degrees, which equates to a centered patch.
  rads = math.pi / 4  # 45 degrees, which is worst case
  bounding_size = math.ceil((size+2*max_shift) * (math.cos(rads) + math.sin(rads)))
  row_center = col_center = round(bounding_size / 2)
  # TODO: either emit a warning, or add a parameter to allow empty corners
  assert bounding_size < min(h, w), "patch size is too large to avoid empty corners after rotation"

  for row, col in coords:
    bounding_patch = Image.fromarray(extract_patch(im, row, col, bounding_size))  # PIL for rotation

    # rotations
    for theta in np.linspace(0, 180, rotations+1, dtype=int):  # always include 0 degrees
      rotated_patch = np.asarray(bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy

      # random translations
      shifts = [gen_random_translation(h, w, row, col, max_shift) for _ in range(rotations)]
      for row_shift, col_shift in [(0, 0)] + shifts:  # always include 0 shift
        patch = extract_patch(rotated_patch, row_center + row_shift, col_center + col_shift, size)
        patch = patch.astype(orig_dtype)  # convert back to original data type

        # sample from a Bernoulli distribution with probability `p`
        if np.random.binomial(1, p):
          yield patch, row, col, theta, row_shift, col_shift


def save_patch(patch, path, lab, case, region, row, col, rotation, row_shift, col_shift, suffix="",
    ext="png"):
  """Save an image patch with an appropriate filename.

  The filename should contain all of the information needed to be able
  to extract the same patch again.

  Args:
    patch: An image patch stored as a NumPy array of shape
      (size, size, c).
    path: A string path to the folder in which to store the image.
    lab: An integer laboratory number from which the patch originated.
    case: An integer case number from which the patch originated.
    region: An integer region number from which the patch originated.
    row: An integer row number at which the patch is centered, before
      rotation and translation.
    col: An integer column number at which the patch is centered, before
      rotation and translation.
    rotation: Integer degrees of rotation.
    row_shift: Integer relative row translation of a patch that was
      centered at (row, col) and then rotated.
    col_shift: Integer relative column translation of a patch that was
      centered at (row, col) and then rotated.
    suffix: An optional string suffix to append to the filename, before
      the file extension.
    ext: A string file extension.
  """
  # lab is a single digit, case and region are two digits with padding if needed
  # TODO: extract filename generation and arg extraction into separate functions
  filename = f"{lab}_{case}_{region}_{row}_{col}_{rotation}_{row_shift}_{col_shift}{suffix}.{ext}"
  file_path = os.path.join(path, filename)
  # NOTE: the subsampling and quality parameters will only affect jpeg images
  Image.fromarray(patch).save(file_path, subsampling=0, quality=100)


def preprocess(images_path, labels_path, base_save_path, train_size, patch_size, rotations_train,
    rotations_val, translations_train, translations_val, max_shift, stride_train, stride_val,
    overlap_threshold, p_train, p_val, fp_path=None, model=None, model_name=None,
    pred_threshold=None, fp_rotations=None, fp_translations=None, seed=None):
  """Generate a mitosis detection patch dataset.

  This generates train/val datasets of mitosis/normal image patches for
  the mitosis detection problem.  The mitosis patches will be extracted
  with centers at the given coordinates, along with random rotations
  and translations from those coordinates.  Normal patches will be
  extracted in a sliding window fashion with the given stride, possibly
  overlapping with mitosis patches up to some given threshold, and
  optionally with false-positive oversampling.  The train/val split will
  be performed on overall cases, stratified by lab.  I.e., the cases
  from each lab will be separately split into training and validation
  sets, and then the associated sets will be combined at the end.  In
  order to support adversarial training, the generated patch filenames
  will each contain information about the laboratory and case from which
  the patch originated.

  Args:
    images_path: Path to folder that contains the mitosis training
      images.
    labels_path: Path to folder that contains the mitosis training
      labels.
    base_save_path: Path to folder in which to write the folders of
      output patches
    train_size: Decimal percentage of data to include in the training
      set during the train/val split.
    patch_size: An integer size of the square patch to extract.
    rotations_train: Integer number of evenly-spaced rotation augmented
      patches to extract for each mitosis in the training set, in
      addition to the centered mitosis patch.
    rotations_val: Integer number of evenly-spaced rotation augmented
      patches to extract for each mitosis in the validation set, in
      addition to the centered mitosis patch.
    translations_train: Integer number of random translation augmented
      patches to extract for each rotated mitosis patch in the training
      set, in addition to the centered rotated mitosis patch.
    translations_val: Integer number of random translation augmented
      patches to extract for each rotated mitosis patch in the
      validation set, in addition to the centered rotated mitosis patch.
    max_shift: Integer upper bound on the spatial shift range for
      the random translations.
    stride_train: An integer number of pixels by which to shift in the
      sliding window for normal patches in the training set.
    stride_val: An integer number of pixels by which to shift in the
      sliding window for normal patches in the validation set.
    overlap_threshold: Decimal inclusive upper bound on the percentage
      of overlap of normal patches with mitosis patches.
    p_train: A decimal probability of sampling each normal patch
      in the training set.
    p_val: A decimal probability of sampling each normal patch
      in the validation set.
    fp_path: Optional path to a folder that contains false-positive
      coordinates, which will be used instead of the model for
      false-positive oversampling.
    model: Optional Keras Model to use for false-positive oversampling.
    model_name: String indicating the model being used, which is used
      for determining the correct normalization.  TODO: replace this
    pred_threshold: Decimal threshold over which the patch is predicted as a
      positive case.
    fp_rotations: Integer number of evenly-spaced rotation augmented
      patches to extract for each false-positive patch in the training
      set, in addition to the centered patch.
    fp_translations: Integer number of random translation
      augmented patches to extract for each rotated false-positive patch
      in the training set, in addition to the centered rotated patch.
    seed: Integer random seed for NumPy.
  """
  # set numpy seed
  np.random.seed(seed)

  # lab info
  lab1 = list(range(1, 24))  # cases 1-23
  lab2 = list(range(24, 49))  # cases 24-48
  lab3 = list(range(49, 74))  # cases 49-73
  labs = {1: lab1, 2: lab2, 3: lab3}

  # generate & save patches
  for lab in range(1, 4):  # 3 labs
    # split cases into train/val sets
    lab_cases = labs.get(lab)
    # TODO: extract this out into a separate function
    train, val = train_test_split(lab_cases, train_size=train_size, test_size=1-train_size,
        random_state=seed)
    train_args = ('train', train, translations_train, rotations_train, p_train, stride_train)
    val_args = ('val', val, translations_val, rotations_val, p_val, stride_val)
    for split_args in [train_args, val_args]:
      # generate samples for this split
      split_name, cases, translations, rotations, p, stride = split_args
      for case in cases:
        case = "{:02d}".format(case)  # reformat case to zero-padded 2-character number
        case_path = os.path.join(images_path, case)
        region_ims = os.listdir(case_path)  # get regions
        for region_im in region_ims:  # a single case may have many available regions
          region, ext = region_im.split('.')  # region number, image file extension
          region_im_path = os.path.join(case_path, region_im)
          im = np.array(Image.open(region_im_path))  # get region image in np.uint8 format
          h, w, c = im.shape
          coords_path = os.path.join(labels_path, case, "{}.csv".format(region))
          if os.path.isfile(coords_path):
            coords = np.loadtxt(coords_path, dtype=np.int64, delimiter=',', ndmin=2)  #, usecols=(0,1))
          else:  # a missing file indicates no mitoses
            coords = []  # no mitoses

          # mitosis samples:
          # save a centered patch, as well as rotations and random translations thereof
          save_path = os.path.join(base_save_path, split_name, "mitosis")
          if not os.path.exists(save_path):
            os.makedirs(save_path)  # create if necessary
          patch_gen = gen_patches(im, coords, patch_size, rotations, translations, max_shift, 1)
          for patch, row, col, rot, row_shift, col_shift in patch_gen:
            save_patch(patch, save_path, lab, case, region, row, col, rot, row_shift, col_shift)

          # normal samples:
          # sample from all possible normal patches
          save_path = os.path.join(base_save_path, split_name, "normal")
          if not os.path.exists(save_path):
            os.makedirs(save_path)  # create if necessary
          mask = create_mask(h, w, coords, patch_size)
          # optional false_positive oversampling
          if fp_path is not None:
            fp_coords_path = os.path.join(fp_path, case, "{}.csv".format(region))
            if os.path.isfile(fp_coords_path):
              fp_coords = np.loadtxt(fp_coords_path, dtype=np.int64, delimiter=',', ndmin=2)
            else:  # a missing file indicates no mitoses
              fp_coords = []  # no mitoses
          elif model is not None and split_name == "train":
            # oversample all false-positive cases in the training set
            normal_coords_gen = gen_normal_coords(mask, patch_size, stride, overlap_threshold)
            fp_coords = gen_fp_coords(im, normal_coords_gen, patch_size, model, model_name,
                pred_threshold)
          else:
            fp_coords = []
          fp_patch_gen = gen_patches(im, fp_coords, patch_size, fp_rotations, fp_translations,
              max_shift, 1)
          for patch, row, col, rot, row_shift, col_shift in fp_patch_gen:
            save_patch(patch, save_path, lab, case, region, row, col, rot, row_shift, col_shift)
          # regular sampling for normal cases
          # NOTE: This may sample the false-positive patches again, but that's fine for now
          normal_coords_gen = gen_normal_coords(mask, patch_size, stride, overlap_threshold)
          patch_gen = gen_patches(im, normal_coords_gen, patch_size, 0, 0, max_shift, p)
          for patch, row, col, rot, row_shift, col_shift in patch_gen:
            save_patch(patch, save_path, lab, case, region, row, col, rot, row_shift, col_shift)


if __name__ == "__main__":
  def check_float_range(x, lb, ub):
    """Argparse utility function for a float type in [lb, ub]."""
    try:
      x = float(x)
    except ValueError as err:
      raise argparse.ArgumentTypeError(str(err))
    if x < lb or x > ub:
      err = "Value should be in [{}, {}]. Got {} instead.".format(lb, ub, x)
      raise argparse.ArgumentTypeError(err)
    return x

  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--images_path",
      default=os.path.join("data", "mitoses", "mitoses_train_image_data"),
      help="path to the mitosis training images (default: %(default)s)")
  parser.add_argument("--labels_path",
      default=os.path.join("data", "mitoses", "mitoses_train_ground_truth"),
      help="path to the mitosis training labels (default: %(default)s)")
  parser.add_argument("--save_path", default=os.path.join("data", "mitoses", "patches"),
      help="path to folder in which to write the folders of output patches (default: %(default)s)")
  parser.add_argument("--train_size", type=lambda x: check_float_range(x, 0, 1), default=0.8,
      help="decimal percentage of data to include in the training set during the train/val split "\
           "(default: %(default)s)")
  parser.add_argument("--patch_size", type=int, default=64,
      help="integer length of the square patches to extract (default: %(default)s)")
  parser.add_argument("--rotations_train", type=int, default=5,
      help="number of evenly-spaced rotation augmented patches to extract for each mitosis in the "\
           "training set, in addition to the centered mitosis patch (default: %(default)s)")
  parser.add_argument("--rotations_val", type=int, default=0,
      help="number of evenly-spaced rotation augmented patches to extract for each mitosis in the "\
           "validation set, in addition to the centered mitosis patch (default: %(default)s)")
  parser.add_argument("--translations_train", type=int, default=5,
      help="number of random translation augmented patches to extract for each rotated mitosis "\
           "patch in the training set, in addition to the centered rotated mitosis patch "\
           "(default: %(default)s)")
  parser.add_argument("--translations_val", type=int, default=0,
      help="number of random translation augmented patches to extract for each rotated mitosis "\
           "patch in the validation set, in addition to the centered rotated mitosis patch "\
           "(default: %(default)s)")
  parser.add_argument("--max_shift", type=int,
      help="upper bound on the spatial shift range for the random translations "\
           "(default: `round(patch_size/4)`)")
  parser.add_argument("--stride_train", type=int,
      help="number of pixels by which to shift in the sliding window for normal patches in the "\
           "training set (default: `patch_size*(3/4)`)")
  parser.add_argument("--stride_val", type=int,
      help="number of pixels by which to shift in the sliding window for normal patches in the "\
           "validation set (default: `patch_size*(3/4)`)")
  parser.add_argument("--overlap_threshold", type=lambda x: check_float_range(x, 0, 1),
      default=0.25, help="decimal inclusive upper bound on the percentage of overlap of normal "\
                         "patches with mitosis patches (default: %(default)s)")
  parser.add_argument("--p_train", type=lambda x: check_float_range(x, 0, 1), default=1,
      help="probability of sampling each normal patch in the training set (default: %(default)s)")
  parser.add_argument("--p_val", type=lambda x: check_float_range(x, 0, 1), default=1,
      help="probability of sampling each normal patch in the validation set (default: %(default)s)")
  parser.add_argument("--fp_path",
      help="path to false-positive locations, which will be used instead of the model "\
           "(default: %(default)s)")
  parser.add_argument("--model_path",
      help="path to a Keras model to use for false-positive oversampling (default: %(default)s)")
  # TODO: replace this with unified normalization flag used here and for training
  parser.add_argument("--model_name",
      help="name of the model being used, which is used for determining the correct normalization "\
           "(default: %(default)s)")
  parser.add_argument("--pred_threshold", type=float, default=0,
      help="threshold over which the patch is predicted as a positive case (default: %(default)s)")
  parser.add_argument("--fp_rotations", type=int, default=5,
      help="number of evenly-spaced rotation augmented patches to extract for each false-positive "\
           "patch in the training set, in addition to the centered patch (default: %(default)s)")
  parser.add_argument("--fp_translations", type=int, default=5,
      help="number of random translation augmented patches to extract for each rotated "\
           "false-positive patch in the training set, in addition to the centered rotated patch "\
           "(default: %(default)s)")
  parser.add_argument("--seed", type=int, help="random seed for numpy (default: %(default)s)")
  args = parser.parse_args()

  # set any other defaults
  if args.max_shift is None:
    args.max_shift = round(args.patch_size/4)

  if args.stride_train is None:
    args.stride_train = round(args.patch_size*(3/4))

  if args.stride_val is None:
    args.stride_val = round(args.patch_size*(3/4))

  # create a random seed if needed
  if args.seed is None:
    args.seed = np.random.randint(1e9)

  # save args to file in save folder
  if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
  with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
    f.write(str(args))

  # copy this script to the base save folder
  shutil.copy2(os.path.realpath(__file__), args.save_path)

  # load model for false-positive oversampling
  if args.model_path is not None:
    model = load_model(args.model_path)
  else:
    model = None

  # preprocess!
  preprocess(images_path=args.images_path, labels_path=args.labels_path,
      base_save_path=args.save_path, train_size=args.train_size, patch_size=args.patch_size,
      rotations_train=args.rotations_train, rotations_val=args.rotations_val,
      translations_train=args.translations_train, translations_val=args.translations_val,
      max_shift=args.max_shift, stride_train=args.stride_train, stride_val=args.stride_val,
      overlap_threshold=args.overlap_threshold, p_train=args.p_train, p_val=args.p_val,
      fp_path=args.fp_path, model=model, model_name=args.model_name,
      pred_threshold=args.pred_threshold, fp_rotations=args.fp_rotations,
      fp_translations=args.fp_translations, seed=args.seed)


# ---
# tests
# TODO: eventually move these to a separate file.
# `py.test preprocess_mitoses.py`

def test_create_mask():
  import pytest

  # create image
  h, w, c = 100, 200, 3
  im = np.random.rand(h, w, c)

  # check mask shape and type
  coords = [(50, 40)]
  size = 32
  mask = create_mask(h, w, coords, size)
  assert mask.shape == (h, w)
  assert mask.dtype == bool

  # row error
  with pytest.raises(AssertionError):
    coords = [(-1, 1)]
    size = 32
    create_mask(h, w, [(-1, 1)], 32)

  # col error
  with pytest.raises(AssertionError):
    coords = [(1, -1)]
    size = 32
    create_mask(h, w, coords, size)

  # size error
  with pytest.raises(AssertionError):
    coords = [(1, 1)]
    size = h+1
    create_mask(h, w, coords, size)

  # another size error
  with pytest.raises(AssertionError):
    coords = [(1, 1)]
    size = w
    create_mask(h, w, coords, size)

  # another size error
  with pytest.raises(AssertionError):
    coords = [(1, 1)]
    size = 1
    create_mask(h, w, coords, size)

  # row, col, size on boundary
  coords = [(0, 0)]
  size = h
  half_size = int(size / 2)
  mask = create_mask(h, w, coords, size)
  correct_mask = np.zeros_like(mask)
  correct_mask[0:half_size, 0:half_size] = 1
  assert np.array_equal(mask, correct_mask)

  # row, col, size on another boundary
  coords = [(h, w)]
  size = h
  half_size = int(size / 2)
  mask = create_mask(h, w, coords, size)
  correct_mask = np.zeros_like(mask)
  correct_mask[-half_size:, -half_size:] = 1
  assert np.array_equal(mask, correct_mask)

  # normal row, col, size
  coords = [(50, 40), (60, 50)]
  size = 32
  half_size = int(size / 2)
  mask = create_mask(h, w, coords, size)
  assert mask.shape == (h, w)
  correct_mask = np.zeros_like(mask)
  for row, col in coords:
    correct_mask[row-half_size:row+half_size, col-half_size:col+half_size] = 1
  assert np.array_equal(mask, correct_mask)

  # normal row, col, size w/ NumPy array
  coords = np.array([(50, 40), (60, 50)])
  size = 32
  half_size = int(size / 2)
  mask = create_mask(h, w, coords, size)
  assert mask.shape == (h, w)
  correct_mask = np.zeros_like(mask)
  for row, col in coords:
    correct_mask[row-half_size:row+half_size, col-half_size:col+half_size] = 1
  assert np.array_equal(mask, correct_mask)

  # row, col, size partially outside bounds
  coords = [(50, 40), (10, 190)]
  size = 32
  half_size = int(size / 2)
  mask = create_mask(h, w, coords, size)
  assert mask.shape == (h, w)
  correct_mask = np.zeros_like(mask)
  for row, col in coords:
    row_lower = max(0, row-half_size)
    row_upper = min(h, row+half_size)
    col_lower = max(0, col-half_size)
    col_upper = min(w, col+half_size)
    correct_mask[row_lower:row_upper, col_lower:col_upper] = 1
  assert np.array_equal(mask, correct_mask)


def test_extract_patch():
  import pytest

  # create image
  h, w, c = 100, 200, 3
  im = np.random.rand(h, w, c)
  im2d = np.random.rand(h, w)

  # row error
  with pytest.raises(AssertionError):
    row, col, size = -1, 1, 32
    extract_patch(im, row, col, size)

  # col error
  with pytest.raises(AssertionError):
    row, col, size = 1, -1, 32
    extract_patch(im, row, col, size)

  # size error
  with pytest.raises(AssertionError):
    row, col, size = 1, 1, h+1
    extract_patch(im, row, col, size)

  # another size error
  with pytest.raises(AssertionError):
    row, col, size = 1, 1, w
    extract_patch(im, row, col, size)

  # another size error
  with pytest.raises(AssertionError):
    row, col, size = 1, 1, 1
    extract_patch(im, row, col, size)

  # row, col, size on boundary
  row, col, size = 0, 0, h
  patch = extract_patch(im, row, col, size)
  patch2d = extract_patch(im2d, row, col, size)
  assert patch.shape == (size, size, c)
  assert patch2d.shape == (size, size)

  # row, col, size on another boundary
  row, col, size = h, w, h
  patch = extract_patch(im, row, col, size)
  patch2d = extract_patch(im2d, row, col, size)
  assert patch.shape == (size, size, c)
  assert patch2d.shape == (size, size)

  # normal row, col, size
  row, col, size = 50, 40, 32
  patch = extract_patch(im, row, col, size)
  assert patch.shape == (size, size, c)
  half_size = int(size / 2)
  correct_patch = im[row-half_size:row+half_size, col-half_size:col+half_size]
  assert np.allclose(patch, correct_patch)

  # row, col, size partially outside bounds
  row, col, size = 10, 190, 32
  patch = extract_patch(im, row, col, size)
  assert patch.shape == (size, size, c)
  half_size = int(size / 2)
  # make sure that the correct patch has actually been extracted
  unpadded_patch = patch[6:,:-6]
  correct_unpadded_patch = im[0:row+half_size, col-half_size:w]
  assert np.array_equal(unpadded_patch, correct_unpadded_patch)


def test_gen_dense_coords():
  import types
  import pytest

  # create image
  h, w, c = 100, 200, 3
  im = np.random.rand(h, w, c)
  size = 32
  stride = size

  # check that it returns a generator object
  assert isinstance(gen_dense_coords(h, w, size,  stride), types.GeneratorType)

  # size error
  with pytest.raises(AssertionError):
    next(gen_dense_coords(h, w, h+1, stride))

  # another size error
  with pytest.raises(AssertionError):
    next(gen_dense_coords(h, w, 1, stride))

  # stride error
  with pytest.raises(AssertionError):
    next(gen_dense_coords(h, w, size, -1))

  # another stride error
  with pytest.raises(AssertionError):
    next(gen_dense_coords(h, w, size, 0))

  # normal
  row, col = next(gen_dense_coords(h, w, size, stride))
  assert 0 <= row <= h
  assert 0 <= col <= w

  # list of coords
  coords = list(gen_dense_coords(h, w, size, stride))
  assert len(coords) > 0

  # check that stride < size produces more coordinates
  coords2 = list(gen_dense_coords(h, w, size, 1))
  assert len(coords2) > len(coords)

  # check for correct centered coordinates
  h = 6
  w = 8
  size = 4
  stride = size - 2
  correct_coords = [(0, 0), (0, 2), (0, 4), (0, 6),
                    (2, 0), (2, 2), (2, 4), (2, 6),
                    (4, 0), (4, 2), (4, 4), (4, 6)]
  coords = list(gen_dense_coords(h, w, size, stride))
  assert coords == correct_coords


def test_gen_normal_coords():
  import types
  import pytest

  # create mask
  h, w = 100, 200
  size = 32
  p = 0.6
  threshold = 0.25
  stride = size
  mask = np.zeros((h, w), dtype=bool)
  mask[0:size, 0:size] = True

  # check that it returns a generator object
  assert isinstance(gen_normal_coords(mask, size, stride, threshold), types.GeneratorType)

  # mask shape error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(np.zeros((h, w, 3)), size, stride, threshold))

  # size error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, h+1, stride, threshold))

  # another size error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, 1, stride, threshold))

  # stride error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, size, -1, threshold))

  # another stride error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, size, 0, threshold))

  # threshold error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, size, stride, -1))

  # another threshold error
  with pytest.raises(AssertionError):
    next(gen_normal_coords(mask, size, stride, 2))

  # normal
  row, col = next(gen_normal_coords(mask, size, stride, threshold))
  assert 0 <= row <= h
  assert 0 <= col <= w

  # list of coords
  coords = list(gen_normal_coords(mask, size, stride, threshold))
  assert len(coords) > 0

  # check for correct coords
  h = 6
  w = 8
  size = 4
  stride = size - 2
  threshold = 0.25
  mask = np.zeros((h, w), dtype=bool)
  mask[0:size, 0:size] = True
  correct_coords = [(0, 6), (2, 6), (4, 4), (4, 6)]
  coords = list(gen_normal_coords(mask, size, stride, threshold))
  assert coords == correct_coords

  # check for situations in which no normal patches should be generated
  # - too much mitosis overlap
  coords = list(gen_normal_coords(np.ones((100, 100)), 100, stride, 0))
  assert len(coords) == 0


def test_gen_random_translation():
  import pytest

  # create image
  h, w, c = 100, 200, 3
  im = np.random.rand(h, w, c)

  # row error
  with pytest.raises(AssertionError):
    row, col, max_shift = -1, 1, 32
    gen_random_translation(h, w, row, col, max_shift)

  # col error
  with pytest.raises(AssertionError):
    row, col, max_shift = 1, -1, 32
    gen_random_translation(h, w, row, col, max_shift)

  # max_shift error
  with pytest.raises(AssertionError):
    row, col, max_shift = 1, 1, -1
    gen_random_translation(h, w, row, col, max_shift)

  # row, col, size on boundary
  row, col, max_shift = 0, 0, 16
  row_shift, col_shift = gen_random_translation(h, w, row, col, max_shift)
  assert 0 <= row + row_shift <= h
  assert 0 <= col + col_shift <= w
  assert abs(row_shift) <= max_shift
  assert abs(col_shift) <= max_shift

  # row, col, size on another boundary
  row, col, max_shift = h, w, h
  row_shift, col_shift = gen_random_translation(h, w, row, col, max_shift)
  assert 0 <= row + row_shift <= h
  assert 0 <= col + col_shift <= w
  assert abs(row_shift) <= max_shift
  assert abs(col_shift) <= max_shift

  # normal row, col, size
  row, col, max_shift = 50, 40, 32
  row_shift, col_shift = gen_random_translation(h, w, row, col, max_shift)
  assert 0 <= row + row_shift <= h
  assert 0 <= col + col_shift <= w
  assert abs(row_shift) <= max_shift
  assert abs(col_shift) <= max_shift

  # check that the function is producing translations, rather than returning
  # the same coordinates
  # we sum the shifts over multiple runs to reduce the probability of a false failure
  for row, col, max_shift in [(0, 0, 16), (h, w, h), (50, 40, 32)]:
    row_shifts = 0
    col_shifts = 0
    for i in range(100):
      row_shift, col_shift = gen_random_translation(h, w, row, col, max_shift)
      row_shifts += abs(row_shift)
      col_shifts += abs(col_shift)
    assert row_shifts > 0
    assert col_shifts > 0


def test_gen_patches():
  import types
  import pytest

  # create image
  h = 100
  w = 200
  c = 3
  im = np.random.rand(h, w, c).astype(np.uint8)
  coords = [(2, 6), (4, 4), (4, 6)]
  size = 4
  rotations = 2
  translations = 2
  max_shift = 2
  p = 1

  # check that it returns a generator object
  patch_gen = gen_patches(im, coords, size, rotations, translations, max_shift, p)
  assert isinstance(patch_gen, types.GeneratorType)

  # size error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, -1, rotations, translations, max_shift, p))

  # another size error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, h+1, rotations, translations, max_shift, p))

  # rotations error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, size, -1, translations, max_shift, p))

  # translations error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, size, rotations, -1, max_shift, p))

  # max shift error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, size, rotations, translations, -1, p))

  # prob error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, size, rotations, translations, max_shift, -1))

  # another prob error
  with pytest.raises(AssertionError):
    next(gen_patches(im, coords, size, rotations, translations, max_shift, 2))

  # normal
  patch_gen = gen_patches(im, coords, size, rotations, translations, max_shift, p)
  assert len(list(patch_gen)) > 0


def test_pil_image_saving(tmpdir):
  # NOTE: pytest will provide a temp directory automatically:
  # https://docs.pytest.org/en/latest/tmpdir.html
  tmpdir = str(tmpdir)
  x = np.random.randint(0, 255, dtype=np.uint8, size=(64,64,3))
  Image.fromarray(x).save(os.path.join(tmpdir, "x1.png"))
  Image.fromarray(x).save(os.path.join(tmpdir, "x2.png"), subsampling=0, quality=100)
  Image.fromarray(x).save(os.path.join(tmpdir, "x1.jpg"))
  Image.fromarray(x).save(os.path.join(tmpdir, "x2.jpg"), subsampling=0, quality=100)
  Image.fromarray(x).save(os.path.join(tmpdir, "x2.jpeg"), subsampling=0, quality=100)

  x1png = np.asarray(Image.open(os.path.join(tmpdir, "x1.png")))
  x2png = np.asarray(Image.open(os.path.join(tmpdir, "x2.png")))
  x1jpg = np.asarray(Image.open(os.path.join(tmpdir, "x1.jpg")))
  x2jpg = np.asarray(Image.open(os.path.join(tmpdir, "x2.jpg")))
  x2jpeg = np.asarray(Image.open(os.path.join(tmpdir, "x2.jpeg")))

  assert np.array_equal(x1png, x2png)
  assert not np.array_equal(x1jpg, x2jpg)
  assert not np.array_equal(x1png, x1jpg)
  assert not np.array_equal(x1png, x2jpg)
  assert np.array_equal(x2jpg, x2jpeg)


def test_gen_patches_extract_patches():
  """ this test wants to show the difference between `extract_patch`
    and `gen_patches`, both of which could be used to generate patches
    from the input images but their content will be a little different
    for the patches on the edge, even if setting the rotation,
    translation, and shift for `gen_patches` to be 0. The differences
    between them are 1) `extract_patch` directly extracts the patch
    from the image; if the patch is located on the edge, some padding
    parts will be added to make it as big as the input size; 2)
    `gen_patches` is more complicated than `extract_patch`. To support
    rotation, translation, and shift, `gen_patches` will firstly extract
    a bigger patch than the input size using `extract_patch`, and then
    extract the target batch from the bigger patch. If the bigger patch
    is located on the edge, it will be added with some padding.
    Considering the above different workflow, 'extract_patch' and
    `gen_patches` will generate different patches when the patch is
    located on some edges. The difference will be their first row/col.
    It is caused by `np.pad(patch, padding, 'reflect')`. When the
    padding size is bigger than the input patch size, the padding
    content will be repeated until reaching the padding size. This
    repeating process will result in the different repeat pattern for
    the patches with different size even if they are extracted by the
    same coordinates from the same image.
  """
  import pytest
  img_org = np.random.randint(low=0, high=256, size=(2000, 2000, 3), dtype=np.uint8)

  # for the patch on the upper edge
  img_extract_patch = extract_patch(img_org, 0, 384, 64)

  img_gen_patches = list(gen_patches(im=img_org, coords=[(0, 384)], size=64, rotations=0,
                             translations=0, max_shift=0, p=1))[0][0]

  assert np.allclose(img_extract_patch[1:,], img_gen_patches[1:,]) # same

  with pytest.raises(AssertionError):
    assert np.allclose(img_extract_patch, img_gen_patches) # different at the first row

  # for the patch in the internal
  img_extract_patch = extract_patch(img_org, 500, 384, 64)

  img_gen_patches = list(gen_patches(im=img_org, coords=[(500, 384)], size=64, rotations=0,
                             translations=0, max_shift=0, p=1))[0][0]

  assert np.allclose(img_extract_patch, img_gen_patches) # same

  # for the patch on the down edge
  img_extract_patch = extract_patch(img_org, 1999, 384, 64)

  img_gen_patches = list(gen_patches(im=img_org, coords=[(1999, 384)], size=64, rotations=0,
                             translations=0, max_shift=0, p=1))[0][0]

  assert np.allclose(img_extract_patch, img_gen_patches) # same

  # for the patch on the left edge
  img_extract_patch = extract_patch(img_org, 384, 0, 64)

  img_gen_patches = list(gen_patches(im=img_org, coords=[(384, 0)], size=64, rotations=0,
                                     translations=0, max_shift=0, p=1))[0][0]

  assert np.allclose(img_extract_patch[:,1:], img_gen_patches[:, 1:])  # same
  with pytest.raises(AssertionError):
    assert np.allclose(img_extract_patch, img_gen_patches) # different at the first col

  # for the patch on the right edge
  img_extract_patch = extract_patch(img_org, 384, 1999, 64)

  img_gen_patches = list(gen_patches(im=img_org, coords=[(384, 1999)], size=64, rotations=0,
                                     translations=0, max_shift=0, p=1))[0][0]

  assert np.allclose(img_extract_patch, img_gen_patches)  # same

