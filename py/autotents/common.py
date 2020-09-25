"""Commonly used definitions."""

import os

import cv2


# This should point to a directory that stores assets not meant for source version control.
# If you want to change to a different directory, this should be the only variable you need to change.
PRIVATE_BASE = '../private'

# (height, width) of the screen that we are building preset against.
# This variable is only used to run analysis on collected samples and should not impact solver.
PRESET_SCREEN_DIM = (2880, 1440)

PUZZLE_SIZES = range(5, 22+1)

# Various colors this game uses, in CV conventional order, that is, (B,G,R).
COLOR_DIGIT_UNSAT = (0x41, 0x4e, 0x7e) # an unsatisfied digit
COLOR_DIGIT_SAT = (0x97, 0xa7, 0xc8)  # a satisfied digit
COLOR_TREE_SHADE = (0x55, 0xc8, 0x87)  # A sample color for tree shade.
COLOR_CELL_BLANK = (0x31, 0x31, 0x34)  # Color of a blank cell.

# We use CCOEFF here as we do want some penalty on mismatched bits
# so that result is spreaded over a wider range so we have finer control using threshold.
TM_METHOD = cv2.TM_CCOEFF_NORMED

# TODO: instead of those thresholds, another way that might worth exploring
# is to gather all tags that has a good score, and see if there are any wrong tags
# and if so, how far away are they from the chosen tag.
# if, say a correct tag gets 0.95 matching score and a wrong tag gets 0.93,
# this might be an indication that we'll need to record the image we are matching
# as a sample to distinguish between two tags.

# threshold used for sampling, this is higher that threshold used for
# recognition as we do want a wider range of samples.
SAMPLE_THRESHOLD = 0.9
# threshold used for recognition.
# a matching result lower than this is considered not confident and may be incorrect.
RECOG_THRESHOLD = 0.85


def private_path(*p):
  """Shorthand for building path to a private asset."""
  return os.path.join(PRIVATE_BASE, *p)


def load_sample_by_name(name,screen_dim=PRESET_SCREEN_DIM):
  """Loads screenshot sample by file name."""
  h, w = screen_dim
  img = cv2.imread(private_path('samples', f'{h}x{w}', name))
  assert img is not None, f'Loaded image is empty, the file might not exist or might be ill-formed.'
  img_h, img_w, _ = img.shape
  assert (img_h, img_w) == screen_dim, \
    f'Image shape mismatched, expected {h}x{w}, got {img_h}x{img_w}.'
  return img


def load_sample(size,screen_dim=PRESET_SCREEN_DIM):
  """Loads screenshot sample of a specific size."""
  return load_sample_by_name(f'sample-{size}x{size}.png')


def find_exact_color(img, color):
  """Calls cv2.inRange restricting to an exact color value."""
  return cv2.inRange(img, color, color)


def rescale_and_match(img, templ_in, tm_method):
  (_,_,w,h) = cv2.boundingRect(img)
  if w == 0 or h == 0:
    return None
  else:
    # try to rescale pattern to match image width (of the bounding rect)
    # we are targeting width here because we can prevent one digit pattern
    # to match with multiple digit ones this way.
    # also because digits tend to vary more in horizontal direction
    # so we are actually eliminating lots of candidates this way.
    templ_in_h, templ_in_w = templ_in.shape
    scale = w / templ_in_w
    templ_h = round(templ_in_h * w / templ_in_w)
    if templ_h > h:
      return None
    templ = cv2.resize(templ_in, (w, templ_h), cv2.INTER_AREA)

  result = cv2.matchTemplate(img, templ, tm_method)
  _, max_val, _, _ = cv2.minMaxLoc(result)
  return max_val


def extract_digits(img, cell_bounds):
  h, w, _ = img.shape
  row_bounds, col_bounds = cell_bounds
  max_cell_side = max(map(lambda x: x[1] - x[0] + 1, row_bounds + col_bounds))
  def extract_digit(row,col):
    return img[row:row+max_cell_side-1,col:col+max_cell_side-1]

  # Suppose first two cells are A and B, we can then find a cell C if we extend
  # difference between A and B but in the other direction.
  # A - (B - A) = 2A - B

  digit_row_start = 2 * row_bounds[0][0] - row_bounds[1][0]
  # col could be negative for 5x5 case.
  digit_col_start = max(0, 2 * col_bounds[0][0] - col_bounds[1][0])

  # digits accompanying every row.
  row_digits = [
    extract_digit(row_lo,digit_col_start)
    for row_lo, _ in row_bounds
  ]
  # same but for columns
  col_digits = [
    extract_digit(digit_row_start,col_lo)
    for col_lo, _ in col_bounds
  ]
  return row_digits, col_digits


def crop_digit_cell(img):
  """Crop a digit sample into 2-color and compact shape.

  This usually done to prepare a sample digit for storage.
  Resulting image (if any) contains at most two colors:
  0 for black and 255 for white.
  """
  result = find_exact_color(img, COLOR_DIGIT_UNSAT)
  (x,y,w,h) = cv2.boundingRect(result)
  if w == 0 or h == 0:
    return None
  return result[y:y+h,x:x+w]

