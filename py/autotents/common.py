"""Commonly used definitions."""

import os

import cv2


# This should point to a directory that stores assets not meant for source version control.
# If you want to change to a different directory, this should be the only variable you need to change.
# TODO: the last sentence is a lie, make sure it is not in the future.
PRIVATE_BASE = '../private'

# Various colors this game uses, in CV conventional order, that is, (B,G,R).
COLOR_DIGIT_UNSAT = (0x41, 0x4e, 0x7e) # an unsatisfied digit
COLOR_DIGIT_SAT = (0x97, 0xa7, 0xc8)  # a satisfied digit
COLOR_TREE_SHADE = (0x55, 0xc8, 0x87)  # A sample color for tree shade.
COLOR_CELL_BLANK = (0x31, 0x31, 0x34)  # Color of a blank cell.



def private_path(*p):
  """Shorthand for building path to a private asset."""
  return os.path.join(PRIVATE_BASE, *p)


def load_sample(size):
  """Loads screenshot sample of a specific size."""
  return cv2.imread(private_path(f'sample-{size}x{size}.png'))


def find_exact_color(img, color):
  """Calls cv2.inRange restricting to an exact color value."""
  return cv2.inRange(img, color, color)
