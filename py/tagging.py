#!/usr/bin/env python3.7
"""The tagging process.

Run this program repeatedly to collect sample of digits and tag them manually.
We want to run this in iterations to minimize the possibility that some untagged digits
are actually similar to each other but both of them end up recorded in the sample.
In addition, doing this in iterations also reduces the amount of manual tagging work
as untagged digits can be removed once recorded digits can match them.
"""


import collections
import functools
import os
import re
import sys
import uuid

import cv2

import autotents.common
import autotents.preset
import autotents.digits


_SAMPLE_FILE_PATTERN = re.compile(r'^.*\.png$', re.IGNORECASE)


def get_all_samples(screen_dim):
  h, w = screen_dim
  retd = collections.defaultdict(list)
  # for this step all samples are participating.
  for fname in os.listdir(autotents.common.private_path('samples', f'{h}x{w}')):
    if _SAMPLE_FILE_PATTERN.match(fname) is None:
      continue
    try:
      img = autotents.common.load_sample_by_name(
        fname,
        autotents.common.PRESET_SCREEN_DIM
      )
      assert img is not None
    except:
      msg = sys.exc_info()[0]
      print(f'Sample {fname} failed to load due to error {msg}, skipping ...')
      continue
    size = autotents.preset.preset.findBoardSize(img, screen_dim)
    if size is None:
      print(f'Skipping sample {fname} as we cannot determine its size ...')
      continue
    retd[size].append((fname, img))
  # returns a sorted list of (size, list of <file name and img instance pair>)
  return sorted(retd.items(), key=lambda x: x[0])


def main_tagging(dry_run=True):
  tagged_samples = autotents.digits.manager.data
  sample_count = functools.reduce(lambda acc, l: acc + len(l), tagged_samples.values(), 0)
  print(f'Loaded {len(tagged_samples)} tags, {sample_count} tagged samples in total.')

  screen_dim = autotents.common.PRESET_SCREEN_DIM
  scr_h, scr_w = screen_dim

  # limit the # of samples stored to disk per function call.
  # for now this is effectively not doing anything but we want to avoid
  # running into a situation that saves too many files at once.
  store_quota = 40
  visit_count = 0
  good_count = 0
  store_path = autotents.common.private_path('digits')
  if not os.path.exists(store_path):
    os.makedirs(store_path)

  for size, samples in get_all_samples(screen_dim):
    if store_quota <= 0:
      break
    print(f'Working on size {size} ...')
    cell_bounds = autotents.preset.preset.getCellBounds(size, screen_dim)
    for fname, img in samples:
      if store_quota <= 0:
        break
      print(f'Processing {fname} ...')
      row_digits, col_digits = autotents.common.extract_digits(img, cell_bounds)
      for digit_img in row_digits + col_digits:
        digit_img_cropped = autotents.common.crop_digit_cell(digit_img)
        if digit_img_cropped is None:
          continue
        visit_count += 1
        # use original image for this step as we want some room around
        # the sample to allow some flexibility.
        best_val, best_tag = autotents.digits.manager.findTag(digit_img)
        if best_val is not None and best_val >= autotents.common.SAMPLE_THRESHOLD:
          good_count += 1
          continue

        nonce = str(uuid.uuid4())
        if best_val is None:
          print(f'Found new sample with no good guesses.')
          fname = f'UNTAGGED_{nonce}.png'
        else:
          print(f'Found new sample with best guess being {best_tag}, with score {best_val}')
          # attach the suspected tag here so it is more convenient when it is actually correct.
          fname = f'UNTAGGED_{best_tag}_{nonce}.png'

        fpath = os.path.join(store_path, fname)
        if dry_run:
          print(f'(Dry run) Saving a sample shaped {digit_img_cropped.shape} to {fpath}...')
        else:
          print(f'Saving a sample shaped {digit_img_cropped.shape} to {fpath}...')
          cv2.imwrite(fpath, digit_img_cropped)
        store_quota -= 1
        if store_quota <= 0:
          break

  print(f'Store quota is now {store_quota}.')
  print(f'Visited {visit_count} samples and {good_count} of them found good matches.')


if __name__ == '__main__':
  main_tagging(dry_run=False)

