#!/usr/bin/env python3.7

import collections
import json

import cv2
import numpy

import autotents.common


def resolve_stat(d, size, threshold=3):
  """
    Given a dict d and an expected # of elements,
    derive a list of row values (or column values) from it.
  """
  hold = None  # or (k, <sub dict>)
  grouping = []
  for k, v in sorted(d.items(), key=lambda x: x[0]):
    if hold is None:
      hold = (k, {k: v})
    else:
      kh, sd = hold
      if k - kh < threshold:
        sd[k] = v
      else:
        grouping.append(sd)
        hold = (k, {k: v})

  if hold is not None:
    grouping.append(hold[1])
    hold = None

  # TODO: given sufficient info we might be able to
  # "fill in the blank" if there are missing elements,
  # but for now it seems good enough to not worry about
  # this issue.
  assert size is None or len(grouping) == size

  # calculate weighted average from grouping elements.
  def ave(sub_dict):
    numer = sum(k * v for k, v in sub_dict.items())
    denom = sum(sub_dict.values())
    return numer / denom

  return map(ave, grouping)


def find_cell_bounds(img, size=None):
  h, w, _ = img.shape
  result = autotents.common.find_exact_color(
    img,
    autotents.common.COLOR_CELL_BLANK
  )

  mk_stat = lambda: collections.defaultdict(lambda: 0)

  row_begins_stat = mk_stat()
  row_ends_stat = mk_stat()

  col_begins_stat = mk_stat()
  col_ends_stat = mk_stat()

  mask = numpy.zeros((h+2,w+2), dtype=numpy.uint8)

  # skip first region encountered, which is likely just the difficulty box
  # on the top right corner.
  first_skipped = False
  for r in range(h):
    for c in range(w):
      if (result[r,c] != 0):
        x,y = c,r
        retval, result, _, rect = cv2.floodFill(result, mask, (x,y), 0)
        rect_x, rect_y, rect_w, rect_h = rect

        if not first_skipped:
          first_skipped = True
          continue

        row_begins_stat[rect_y] += 1
        col_begins_stat[rect_x] += 1

        rect_x_end = rect_x + rect_w - 1
        rect_y_end = rect_y + rect_h - 1
        row_ends_stat[rect_y_end] += 1
        col_ends_stat[rect_x_end] += 1

  def make_bounds(begin_stat, end_stat):
    begin_coords = map(round, resolve_stat(begin_stat, size))
    end_coords = map(round, resolve_stat(end_stat, size))
    return list(map(lambda x,y: (x,y), begin_coords, end_coords))

  row_bounds = make_bounds(row_begins_stat, row_ends_stat)
  col_bounds = make_bounds(col_begins_stat, col_ends_stat)

  if size is None:
    assert len(row_bounds) == len(col_bounds), f'Mismatched bound length {len(row_bounds)} vs {len(col_bounds)}.'

  return row_bounds, col_bounds


# Given that most of the processing time is spent on doing floodFill to figure out cell bounds,
# it makes sense that we have this info pre-processed. In order to achieve so, we must extract size of the board.
# Note that despite regular puzzle shows size info (size x size), daily puzzles do not.
# one potential alternative is to examine an empty cell of the board and see if it's possible to establish size this way
# (assuming that all puzzles are squares)
def main_generate_preset():
  # schema:
  # top level is an Object keyed by screen width and height i.e. "1440x2880"
  # then values are Object keyed by size e.g. "16x16", which is then keyed by "row_bounds" and "col_bounds",
  # which are Arrays whose elements are Arrays of two elements [lo, hi].
  # e.g.:
  # {
  #   "2880x1440": {"16x16": {"row_bounds": [[a,b], [c,d], ...], "col_bounds": [[a,b], [c,d], ...]}}
  # }
  h, w = autotents.common.PRESET_SCREEN_DIM
  cell_bounds_mapping = {}
  for size in autotents.common.PUZZLE_SIZES:
    print(f'Processing {size}x{size} ...')
    img = autotents.common.load_sample(size)
    cell_bounds = find_cell_bounds(img, size)
    row_bounds, col_bounds = cell_bounds
    cell_bounds_mapping[f'{size}x{size}'] = {
      'row_bounds': row_bounds,
      'col_bounds': col_bounds,
    }
  full = {f'{h}x{w}': cell_bounds_mapping}
  print(json.dumps(full,sort_keys=True,separators=(',', ':')))


if __name__ == '__main__':
  main_generate_preset()
