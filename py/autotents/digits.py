"""This module deals with digit sample management and digit recognition."""


import collections
import os
import re


import cv2

import autotents.common


_SAMPLE_FILENAME_PATTEN = re.compile(r'^([^_]+)_.*.png$')


class SampleManager:

  def __init__(self):
    self.load()

  def load(self):
    self.data = collections.defaultdict(list)
    store_path = autotents.common.private_path('digits')
    if not os.path.exists(store_path):
      return

    untagged_count = 0
    for filename in os.listdir(store_path):
      result = _SAMPLE_FILENAME_PATTEN.match(filename)
      if result is None:
        continue
      tag = result.group(1)
      if tag == 'UNTAGGED':
        untagged_count += 1
        continue
      self.data[tag].append(cv2.imread(os.path.join(store_path, filename),cv2.IMREAD_GRAYSCALE))

    if untagged_count:
      print(f'There are {untagged_count} untagged samples.')

  def findTag(self, img_pre, tm_method=autotents.common.TM_METHOD):
    img = autotents.common.find_exact_color(img_pre, autotents.common.COLOR_DIGIT_UNSAT)
    best_val, best_tag = None, None
    for tag, samples in self.data.items():
      for pat in samples:
        val = autotents.common.rescale_and_match(img,pat,tm_method)
        if val is None:
          continue
        if best_val is None or best_val < val:
          best_val, best_tag = val, tag
    return best_val, best_tag


manager = SampleManager()
