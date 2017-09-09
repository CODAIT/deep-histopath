"""Preprocessing - mitosis detection"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def extract_patch(im, row, col, size):
  """Extract patch centered at (row, col).

  If the (row, col) is at the edge of the image, the image will be
  reflected to yield a patch of the desired size.

  Args:
    im: An image stored as a NumPy array of shape (h, w, c).
    row: An integer row number.
    col: An integer col number.
    size: An integer size of the square patch to extract.

  Returns:
    A NumPy array of shape (size, size, c).
  """
  # check that row, col, and size are within the image bounds
  h, w, c = im.shape
  assert 0 <= row <= h, "row is outside of the image height"
  assert 0 <= col <= w, "col is outside of the image width"
  assert 1 < size <= min(h, w), "size must be >1 and within the bounds of the image"

  # (row, col) is the center, so compute upper and lower bounds of patch
  half_size = int(size / 2)
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

  # pad with reflection as needed to yield a patch of the desired size
  padding = ((row_pad_lower, row_pad_upper), (col_pad_lower, col_pad_upper), (0, 0))
  patch_padded = np.pad(patch, padding, 'reflect')

  return patch_padded


def gen_random_translation(im, row, col, max_shift):
  """Generate (row, col) random translation coordinates.

  Ensures that the coordinates are within the bounds of the image.

  Args:
    im: An image stored as a NumPy array of shape (h, w, c).
    row: An integer row number.
    col: An integer col number.
    size: An integer size of the square patch to extract.

  Returns:
    New (row_shifted, col_shifted) integer coordinates.
  """
  # check that row, col, and size are within the image bounds
  h, w, c = im.shape
  assert 0 <= row <= h, "row is outside of the image height"
  assert 0 <= col <= w, "col is outside of the image width"
  assert max_shift > 0, "max_shift must be > 0"

  row_shifted = min(max(0, row + np.random.randint(-max_shift, max_shift)), h)
  col_shifted = min(max(0, col + np.random.randint(-max_shift, max_shift)), w)
  return row_shifted, col_shifted


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
    half_size = int(size / 2)
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


def gen_normal_coords(mask, size, threshold, num_tries=10000):
  """Generate (row, col) coordinates for normal patches.

  This generates coordinates for normal patches that may overlap with
  mitosis patches up to `threshold` percentage.

  Args:
    mask: A binary mask, indicating where the mitosis patches are
      located, of the same height and width as the region image.
    size: An integer size of the square patch to extract.
    threshold = A decimal inclusive upper bound on the percentage of
      overlap.
    num_tries = Integer upper bound on sampling to prevent an infinite
      loop.

  Returns:
    (row, col) coordinates of a normal patch.
  """
  # check that size is within the mask bounds
  assert np.ndim(mask) == 2, "mask must be of shape (h, w)"
  h, w = mask.shape
  assert 1 < size <= min(h, w), "size must be >1 and within the bounds of the image"
  assert 0 <= threshold <= 1, "threshold must be a valid decimal percentage"

  #while True:
  for _ in range(num_tries):  # upper bound on sampling to prevent infinite loop
    # uniformly sample (row, col) coordinates from the image
    row = np.random.randint(0, h)
    col = np.random.randint(0, w)

    # return (row, col) if the overlap with mitosis patches is <= threshold
    mask_patch = np.squeeze(extract_patch(np.atleast_3d(mask), row, col, size))
    if np.mean(mask_patch) <= threshold:
      return row, col

  else:  # raise error if no normal patches found
    raise AssertionError("no normal patches found after {} tries".format(num_tries))


def save_patch(patch, path, lab, case, region, row, col, suffix="", ext="jpg"):
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
    row: An integer row number at which the patch is centered.
    col: An integer column number at which the patch is centered.
    suffix: An optional string suffix to append to the filename, before
      the file extension.
    ext: A string file extension.
  """
  # lab is a single digit, case and region are two digits with padding if needed
  filename = f"{lab}_{case}_{region}_{row}_{col}_{suffix}.{ext}"
  file_path = os.path.join(path, filename)
  Image.fromarray(patch).save(file_path)


def preprocess(images_path, labels_path, base_save_path, train_size, patch_size,
    num_rand_translations, max_shift, num_normal, overlap_threshold, seed=None):
  """Generate a mitosis detection patch dataset.

  This generates train/val datasets of mitosis/normal image patches for
  the mitosis detection problem.  The mitosis patches will be extracted
  with centers at the given coordinates, along with random translations
  from those coordinates.  Normal patches will be extracted from areas
  outside of the mitosis patches, except for some small overlaps.  To
  support adversarial training, the generated patch filenames will each
  contain information about the laboratory and case from which the
  patch originated.

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
    num_rand_translations: Integer number of random translation
      augmented patches to extract for each mitosis, in addition to the
      centered mitosis patch.
    max_shift: Integer upper bound on the spatial shift range for the
      random translations.
    num_normal: Integer number of normal patches to extract for each
      mitosis.
    overlap_threshold: Decimal inclusive upper bound on the percentage
      of overlap of normal patches with mitosis patches.
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
    lab_cases = labs.get(lab)
    train, val = train_test_split(lab_cases, train_size=train_size, test_size=1-train_size,
        random_state=seed)
    for stage, cases in [('train', train), ('val', val)]:
      for case in cases:
        case = "{:02d}".format(case)  # reformat case to zero-padded 2-character number
        case_path = os.path.join(images_path, case)
        region_ims = os.listdir(case_path)  # get regions
        for region_im in region_ims:
          region, ext = region_im.split('.')  # region number, image file extension
          region_im_path = os.path.join(case_path, region_im)
          im = np.array(Image.open(region_im_path))  # get region image
          h, w, c = im.shape
          coords_path = os.path.join(labels_path, case, "{}.csv".format(region))
          if os.path.isfile(coords_path):  # a missing file indicates no mitoses
            coords = np.loadtxt(coords_path, dtype=np.int64, delimiter=',', ndmin=2)
          else:
            coords = []  # no mitoses

          # mitosis samples:
          save_path = os.path.join(base_save_path, stage, "mitosis")
          if not os.path.exists(save_path):
            os.makedirs(save_path)  # create if necessary
          for row, col in coords:
            # centered mitosis patch
            patch = extract_patch(im, row, col, patch_size)
            save_patch(patch, save_path, lab, case, region, row, col)
            # mitosis random translations
            for i in range(num_rand_translations):
              row_shifted, col_shifted = gen_random_translation(im, row, col, max_shift)
              patch = extract_patch(im, row_shifted, col_shifted, patch_size)
              save_patch(patch, save_path, lab, case, region, row_shifted, col_shifted,
                  "shifted-from-{}-{}".format(row, col))

          # normal samples:
          save_path = os.path.join(base_save_path, stage, "normal")
          if not os.path.exists(save_path):
            os.makedirs(save_path)  # create if necessary
          mask = create_mask(h, w, coords, patch_size)
          for i in range(num_normal):
            row, col = gen_normal_coords(mask, patch_size, overlap_threshold)
            patch = extract_patch(im, row, col, patch_size)
            save_patch(patch, save_path, lab, case, region, row, col)


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
  parser.add_argument("--base_path", default=os.path.join("data", "mitoses"),
      help="base path for mitosis datasets (default: %(default)s)")
  parser.add_argument("--images_folder", default="mitoses_train_image_data",
      help="folder within `base_path` that contains the mitosis training images "\
           "(default: %(default)s)")
  parser.add_argument("--labels_folder", default="mitoses_train_ground_truth",
      help="folder within `base_path` that contains the mitosis training labels "\
           "(default: %(default)s)")
  parser.add_argument("--save_folder", default="patches",
      help="folder within `base_path` in which to write the folders of output patches "\
           "(default: %(default)s)")
  parser.add_argument("--train_size", type=lambda x: check_float_range(x, 0, 1), default=0.8,
      help="decimal percentage of data to include in the training set during the train/val split "\
           "(default: %(default)s)")
  parser.add_argument("--patch_size", type=int, default=64,
      help="integer length of the square patches to extract (default: %(default)s)")
  parser.add_argument("--num_rand_translations", type=int, default=10,
      help="number of random translation augmented patches to extract for each mitosis, in "\
           "addition to the centered mitosis patch (default: %(default)s)")
  parser.add_argument("--max_shift", type=int,
      help="upper bound on the spatial shift range for the random translations "\
           "(default: `int(patch_size/4)`)")
  parser.add_argument("--num_normal", type=int, help="number of normal patches to extract from "\
      "each region (default: `num_rand_translations+1`)")
  parser.add_argument("--overlap_threshold", type=lambda x: check_float_range(x, 0, 1),
      default=0.25, help="decimal inclusive upper bound on the percentage of overlap of normal "\
                         "patches with mitosis patches (default: %(default)s)")
  parser.add_argument("--seed", type=int, help="random seed for numpy (default: %(default)s)")
  args = parser.parse_args()

  # set any other defaults
  images_path = os.path.join(args.base_path, args.images_folder)
  labels_path = os.path.join(args.base_path, args.labels_folder)
  base_save_path = os.path.join(args.base_path, args.save_folder)
  max_shift = args.max_shift if args.max_shift is not None else int(args.patch_size/4)
  num_normal = args.num_normal if args.num_normal is not None else args.num_rand_translations + 1

  # preprocess!
  preprocess(images_path, labels_path, base_save_path, args.train_size, args.patch_size,
      args.num_rand_translations, max_shift, num_normal, args.overlap_threshold, args.seed)


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
    create_mask(h, w, coords, size)

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
  assert patch.shape == (size, size, c)

  # row, col, size on another boundary
  row, col, size = h, w, h
  patch = extract_patch(im, row, col, size)
  assert patch.shape == (size, size, c)

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


def test_gen_normal_coords():
  import pytest
  # create mask
  h, w = 100, 200
  size = 32
  threshold = 0.25
  mask = np.zeros((h, w), dtype=bool)
  mask[0:size, 0:size] = True

  # mask shape error
  with pytest.raises(AssertionError):
    gen_normal_coords(np.zeros((h, w, 3)), size, threshold)

  # size error
  with pytest.raises(AssertionError):
    gen_normal_coords(mask, h+2, threshold)

  # another size error
  with pytest.raises(AssertionError):
    gen_normal_coords(mask, 1, threshold)

  # threshold error
  with pytest.raises(AssertionError):
    gen_normal_coords(mask, size, -1)

  # another threshold error
  with pytest.raises(AssertionError):
    gen_normal_coords(mask, size, 2)

  # check for failure if no normal patches can be found
  with pytest.raises(AssertionError):
    gen_normal_coords(np.ones((100, 100)), 100, 0)

  # normal
  row, col = gen_normal_coords(mask, size, threshold)
  assert 0 <= row <= h
  assert 0 <= col <= w


def test_gen_random_translation():
  import pytest
  # create image
  h, w, c = 100, 200, 3
  im = np.random.rand(h, w, c)

  # row error
  with pytest.raises(AssertionError):
    row, col, max_shift = -1, 1, 32
    gen_random_translation(im, row, col, max_shift)

  # col error
  with pytest.raises(AssertionError):
    row, col, max_shift = 1, -1, 32
    gen_random_translation(im, row, col, max_shift)

  # max_shift error
  with pytest.raises(AssertionError):
    row, col, max_shift = 1, 1, -1
    gen_random_translation(im, row, col, max_shift)

  # another max_shift error
  with pytest.raises(AssertionError):
    row, col, max_shift = 1, 1, 0
    gen_random_translation(im, row, col, max_shift)

  # row, col, size on boundary
  row, col, max_shift = 0, 0, 16
  row_shifted, col_shifted = gen_random_translation(im, row, col, max_shift)
  assert 0 <= row_shifted <= h
  assert 0 <= col_shifted <= w
  assert abs(row - row_shifted) <= max_shift
  assert abs(col - col_shifted) <= max_shift

  # row, col, size on another boundary
  row, col, max_shift = h, w, h
  row_shifted, col_shifted = gen_random_translation(im, row, col, max_shift)
  assert 0 <= row_shifted <= h
  assert 0 <= col_shifted <= w
  assert abs(row - row_shifted) <= max_shift
  assert abs(col - col_shifted) <= max_shift

  # normal row, col, size
  row, col, max_shift = 50, 40, 32
  row_shifted, col_shifted = gen_random_translation(im, row, col, max_shift)
  assert 0 <= row_shifted <= h
  assert 0 <= col_shifted <= w
  assert abs(row - row_shifted) <= max_shift
  assert abs(col - col_shifted) <= max_shift

  # check that the function is producing translations, rather than returning
  # the same coordinates
  # we sum the shifts over multiple runs to reduce the probability of a false failure
  for row, col, max_shift in [(0, 0, 16), (h, w, h), (50, 40, 32)]:
    row_shifts = 0
    col_shifts = 0
    for i in range(100):
      row_shifted, col_shifted = gen_random_translation(im, row, col, max_shift)
      row_shifts += abs(row - row_shifted)
      col_shifts += abs(col - col_shifted)
    assert row_shifts > 0
    assert col_shifts > 0

