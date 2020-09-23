import json
import os
import re

import cv2
import numpy

import autotents.common


_RE_RAW_SIZE = re.compile(r'^(\d+)x\1$')


def _to_side_length_set(bounds):
    return { x[1] - x[0] + 1 for x in bounds }


class Preset:

  def __init__(self):
    self.load()

  def location(self):
    return autotents.common.private_path('preset.json')

  def load(self):
    loc = self.location()
    if os.path.exists(loc):
      print('Loading preset ...')
      with open(loc, 'r') as f:
        self.data = json.load(f)
    else:
      self.data = {}

  def save(self):
    print('Saving preset ...')
    with open(self.location(), 'w') as f:
      json.dump(self.data,fp=f,sort_keys=True,separators=(',', ':'))

  def register(self, screen_dim_desc, cell_bounds_mapping):
    self.data[screen_dim_desc] = cell_bounds_mapping
    self.save()

  def buildSideLengthRevMap(self, screen_dim):
    h, w = screen_dim
    # Build reverse map from side length of a blank cell to size (# of cells in row or col)
    ret = {}
    for size_raw, v in self.data[f'{h}x{w}'].items():
      size = int(_RE_RAW_SIZE.match(size_raw).group(1))
      row_bounds = _to_side_length_set(v['row_bounds'])
      col_bounds = _to_side_length_set(v['col_bounds'])
      all_bounds = set.union(row_bounds, col_bounds)
      for x in all_bounds:
        assert x not in ret, 'Side length is ambiguous.'
        ret[x] = size
    return ret

  def findBoardSize(self, img, screen_dim):
    h, w, _ = img.shape
    result = autotents.common.find_exact_color(img, autotents.common.COLOR_CELL_BLANK)
    mask = numpy.zeros((h+2,w+2), dtype=numpy.uint8)
    side_length_rev_map = self.buildSideLengthRevMap(screen_dim)
    # now we just need one empty cell for this to work,
    # we can just search inside bounding rect and
    # find the last empty cell so that we don't need to
    # skip first box and then look at many filler lines.
    r_x, r_y, r_w, r_h = cv2.boundingRect(result)
    for r in reversed(range(r_y,r_y+r_h)):
      for c in reversed(range(r_x,r_x+r_w)):
        if (result[r,c] != 0):
          x,y = c,r
          retval, result, _, rect = cv2.floodFill(result, mask, (x,y), 0)
          _, _, rect_w, _ = rect
          if rect_w in side_length_rev_map:
            return side_length_rev_map[rect_w]
          else:
            return None


preset = Preset()
