"""Commonly used definitions."""

import os

import cv2


# This should point to a directory that stores assets not meant for source version control.
# If you want to change to a different directory, this should be the only variable you need to change.
# TODO: the last sentence is a lie, make sure it is not in the future.
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



def private_path(*p):
  """Shorthand for building path to a private asset."""
  return os.path.join(PRIVATE_BASE, *p)


def load_sample(size,screen_dim=PRESET_SCREEN_DIM):
  """Loads screenshot sample of a specific size."""
  h, w = screen_dim
  img = cv2.imread(private_path('samples', f'{h}x{w}', f'sample-{size}x{size}.png'))
  assert img, f'Loaded image is empty, the file might not exist or might be ill-formed.'
  img_h, img_w, _ = img.shape
  assert (img_h, img_w) == screen_dim, \
    f'Image shape mismatched, expected {h}x{w}, got {img_h}x{img_w}.'
  return img


def find_exact_color(img, color):
  """Calls cv2.inRange restricting to an exact color value."""
  return cv2.inRange(img, color, color)
