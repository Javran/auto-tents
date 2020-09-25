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

  def findTagGray(self, img, tm_method=autotents.common.TM_METHOD):
    # first round: collect pairs that are better than a threshold.
    good_values = []
    for tag, samples in self.data.items():
      for pat in samples:
        val = autotents.common.rescale_and_match(img,pat,tm_method)
        if val is None or val < autotents.common.RECOG_THRESHOLD:
          continue
        good_values.append((val, tag))
    if not len(good_values):
      return None, None, None
    good_values = sorted(good_values, key=lambda x: x[0], reverse=True)
    best_val, best_tag = good_values[0]
    # find tags that are considered good by threshold but does not actually match
    # with the best tag, those are "competitors" that could potentially lead to inaccurate results.
    competitors = [p for p in filter(lambda p: p[1] != best_tag, good_values)]
    if not len(competitors):
      competing_factor = None
    else:
      # the competing factor measures "the best wrong match", the lower this value gets,
      # the more likely that we get a wrong result.
      # for now we haven't find cases where this value is not None, so
      # we always treat cases that this is not None as an alerting result and try to sample it if possible.
      print(f't: {best_tag}, {best_val} competitiors: {competitors}')
      competing_factor = best_val - competitors[0][0]
      assert competing_factor > 0
    return best_val, best_tag, competing_factor

  def findTag(self, img_pre, tm_method=autotents.common.TM_METHOD):
    img = autotents.common.find_exact_color(img_pre, autotents.common.COLOR_DIGIT_UNSAT)
    return self.findTagGray(img, tm_method)

  def cleanUpUntagged(self):
    """Remove UNTAGGED sample if we can now find a good match."""
    store_path = autotents.common.private_path('digits')
    for filename in os.listdir(store_path):
      result = _SAMPLE_FILENAME_PATTEN.match(filename)
      if result is None:
        continue
      tag = result.group(1)
      if tag != 'UNTAGGED':
        continue

      full_file_path = os.path.join(store_path, filename)
      img_pre = cv2.imread(full_file_path,cv2.IMREAD_GRAYSCALE)
      padding = 5
      img = cv2.copyMakeBorder(
        img_pre,
        padding, padding, padding, padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0)

      best_val, best_tag, competing_factor = self.findTagGray(img, autotents.common.TM_METHOD)
      if best_val is not None and best_val >= autotents.common.RECOG_THRESHOLD and competing_factor is None:
        print(f'Removing {filename} as it achieves {best_val} with tag {best_tag} ...')
        os.remove(full_file_path)


manager = SampleManager()
