#!/usr/bin/env python3.7
"""
  This program is assumed to be executed in the directory where it lives in,
  so all file paths are relative to that.
"""

import cv2
import functools
import math
import numpy as np
from matplotlib import pyplot
import collections
import os
import uuid
import re
import functools
import json
import subprocess
import io
import tempfile
import time
import random
import autotents.common


def find_and_mark_matches(img, result, pat_dims, threshold):
  """Find and mark matching places given a result of matchTemplate.
  """
  img_marked = img.copy()
  h, w = result.shape
  pat_h, pat_w = pat_dims
  for r in range(h):
    for c in range(w):
      if (result[r,c] > threshold):
        print(r,c, result[r,c])
        top_left = (c,r)
        bottom_right = (c + pat_w, r + pat_h)
        cv2.rectangle(img_marked, top_left, bottom_right, 255, 2)
  return img_marked


def scale_pattern(pat_orig, target_width):
  pat_orig_h, pat_orig_w = pat_orig.shape[0], pat_orig.shape[1]
  scale = target_width / pat_orig_w
  pat_h = round(pat_orig_h * scale)
  pat_w = round(pat_orig_w * scale)
  return cv2.resize(pat_orig, (pat_w, pat_h), cv2.INTER_AREA)


def optimize_pattern_width(pat_orig, img):
  eval_count = 0

  @functools.lru_cache()
  def evaluate_width(width):
    nonlocal eval_count
    pat = scale_pattern(pat_orig, width)
    pat_w, pat_h, _ = pat.shape
    result = cv2.matchTemplate(img,pat,tm_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    eval_count += 1
    return max_val

  # search within this range, with decreasing steps per iteration until
  # we reach a local maxima
  min_width, max_width = 30, 220
  step = 16
  candidates = set(range(min_width, max_width, step))

  while True:
    sorted_candidates = sorted(candidates,key=evaluate_width,reverse=True)
    # Only top few candidates survive.
    keep = max(1, math.floor(len(sorted_candidates) * 0.1))
    candidates = sorted_candidates[:keep]
    step //= 2
    if not step:
      break
    # candidate expansion for next iteration.
    candidates = {
      y
      for x in candidates
      for y in [x-step, x, x+step]
      if min_width <= y <= max_width
    }

  # note that here candidates are sorted
  best_target_width = candidates[0]
  print(f'Best target width is: {best_target_width}, evaluations: {eval_count}')
  return best_target_width


def resample_pattern_from_image(pat_orig, img):
  best_target_width = optimize_pattern_width(pat_orig, img)
  pat = scale_pattern(pat_orig, best_target_width)
  pat_h, pat_w, _ = pat.shape
  result = cv2.matchTemplate(img,pat,tm_method)
  _, _, _, max_loc = cv2.minMaxLoc(result)
  c, r = max_loc
  return img[r:r+pat_h,c:c+pat_w]


def subplot_gray(num, img, title):
  pyplot.subplot(num), pyplot.imshow(img,cmap = 'gray')
  pyplot.title(title), pyplot.xticks([]), pyplot.yticks([])


def subplot_color(num, img, title):
  pyplot.subplot(num), pyplot.imshow(img[:,:,[2,1,0]])
  pyplot.title(title), pyplot.xticks([]), pyplot.yticks([])


def main_all_samples():
  pat_orig = cv2.imread('../sample/tree-sample.png')
  for i in autotents.common.PUZZLE_SIZES:
    img = load_sample(i)
    target_width = optimize_pattern_width(pat_orig, img)
    print(f'{i}: {target_width}')


def main_experiment():
  size = 22
  img = load_sample(size)
  h, w, _ = img.shape

  cell_bounds =  find_cell_bounds(img, size)
  row_bounds, col_bounds = cell_bounds

  cells = [ [ None for _ in range(size) ] for _ in range(size)]
  for r, (row_lo, row_hi) in enumerate(row_bounds):
    for c, (col_lo, col_hi) in enumerate(col_bounds):
      cells[r][c] = img[row_lo:row_hi+1, col_lo:col_hi+1]

  def find_tree(cell_img):
    result = cv2.inRange(cell_img, color_shade, color_shade)
    (_,_,w,h) = cv2.boundingRect(result)
    if w != 0 and h != 0:
      color = 0xFF
    else:
      color = 0
    return np.full((4,4), color)

  recombined = np.concatenate([ np.concatenate(row, axis=1) for row in cells ], axis=0)

  cell_results_recombined = np.concatenate([
    np.concatenate([ find_tree(c) for c in row], axis=1) for row in cells
  ], axis=0)

  max_cell_side = max(map(lambda x: x[1] - x[0] + 1, row_bounds + col_bounds))
  side_length_for_display = math.ceil(max_cell_side * 1.1)

  def padding_digit_img(dg_img):
    if dg_img is None:
      return np.full((side_length_for_display, side_length_for_display), 0x7F)

    h, w = dg_img.shape
    top = math.floor((side_length_for_display - h) / 2)
    bottom = side_length_for_display - top - h
    left = math.floor((side_length_for_display - w) / 2)
    right =  side_length_for_display - left - w
    return cv2.copyMakeBorder(dg_img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0x7F)

  # digits accompanying every row and col.
  row_digits, col_digits = extract_digits(img, cell_bounds)

  row_digit_templs = [ crop_digit_cell(d) for d in row_digits ]
  col_digit_templs = [ crop_digit_cell(d) for d in col_digits ]

  def debug_cross_compare(digits, digit_templs):
    for dg_img_pre in digits:
      dg_img = cv2.inRange(dg_img_pre, color_unsat, color_unsat)
      line = []
      for templ in digit_templs:
        if templ is None:
          line.append('------')
          continue
        max_val = rescale_and_match(dg_img,templ,tm_method)
        if max_val is None:
          line.append('------')
          continue

        line.append(f'{max_val:.4f}')
      print(', '.join(line))

  print('Mat for row digits:')
  debug_cross_compare(row_digits, row_digit_templs)
  print('Mat for col digits:')
  debug_cross_compare(col_digits, col_digit_templs)

  digits = np.concatenate(
    [
      np.concatenate([padding_digit_img(x) for x in row_digit_templs], axis=1),
      np.concatenate([padding_digit_img(x) for x in col_digit_templs], axis=1),
    ])

  # digit sample extraction steps (for each single cell image):
  # - cv2.inRange to extract shape of the digit
  # - cv2.boundingRect to find the bounding rectangle
  # - crop it and save it as image.

  show = True
  if show:
    pyplot.figure().canvas.set_window_title('@dev')
    subplot_color(221, img, 'origin')
    subplot_color(222, recombined, 'extracted')
    subplot_gray(223, digits, 'digits')
    subplot_gray(224, cell_results_recombined, 'find tree')
    pyplot.show()


# Here we focus on two numbers:
# - what is the worst match inside the same tag (in-tag min),
#   this measures how "spreaded" are those samples.
# - what is the best match with other different tags (cross-tag min),
#   this one is arguably the more valuable info of the two,
#   as it provides some guidance on how to set the "it's a good match" threshold.
def main_sample_analysis():
  """Analysis of collected samples."""
  tagged_samples = load_samples()
  flat_samples = sum(
    [[(tag, i, s)] for tag, samples in tagged_samples.items() for i, s in enumerate(samples)],
    []
  )
  print(len(flat_samples))
  # stores match in results[tag0, i0][tag1, i1]
  results = collections.defaultdict(dict)
  for (tag0, i0, s_img_pre) in flat_samples:
    padding = 5
    # Apply padding in all directions, this is to:
    # (1) simulate the situation that we need to match a pattern
    # in an image that contains some extra empty parts.
    # (2) allow some flexibility for matchTemplate
    s_img = cv2.copyMakeBorder(
      s_img_pre,
      padding, padding, padding, padding,
      borderType=cv2.BORDER_CONSTANT,
      value=0)

    for (tag1, i1, s_pat) in flat_samples:
      val = rescale_and_match(s_img,s_pat,tm_method)
      if val is None:
        continue
      results[tag0,i0][tag1,i1] = val

  def minMaxWithoutOne(xs_pre):
    xs = [ x for x in xs_pre if x < 1 ]
    if len(xs):
      return min(xs), max(xs)
    else:
      return None

  in_tag_min, cross_tag_max = None, None
  for tag, samples in tagged_samples.items():
    print(f'Tag {tag} has {len(samples)} samples.')
    l = range(len(samples))
    vals = [
      results[tag, i][tag, j]
      for i in l
      for j in l
      if (tag, j) in results[tag, i]
    ]
    min_max = minMaxWithoutOne(vals)
    print(f'  stats in-tag: min_val, max_val = {min_max}')
    if min_max is not None:
      if in_tag_min is None or min_max[0] < in_tag_min:
        in_tag_min = min_max[0]
    vals = [
      val
      for (tag0, _), d in results.items()
      if tag0 == tag
      for (tag1, _), val in d.items()
      if tag1 != tag
    ]
    min_max = minMaxWithoutOne(vals)
    if min_max is not None:
      if cross_tag_max is None or min_max[1] > cross_tag_max:
        cross_tag_max = min_max[1]
    print(f'  stats cross-tag: min_val, max_val = {min_max}')
  print(f'in-tag min: {in_tag_min}, cross-tag max: {cross_tag_max}')
  # for now the result is:
  # in-tag min: 0.6574000716209412, cross-tag max: 0.7616900205612183
  # so I guess 0.85 could be a decent threshold to use.


if __name__ == '__main__':
  # main_experiment()
  # main_tagging()
  # main_generate_preset()
  # main_recognize_and_solve_board()
  # main_sample_analysis()
  pass
