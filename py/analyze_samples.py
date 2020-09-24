#!/usr/bin/env python3.7

import collections

import cv2

import autotents.common
import autotents.digits

# Here we focus on two numbers:
# - what is the worst match inside the same tag (in-tag min),
#   this measures how "spreaded" are those samples.
# - what is the best match with other different tags (cross-tag min),
#   this one is arguably the more valuable info of the two,
#   as it provides some guidance on how to set the "it's a good match" threshold.
def main_analyze_samples():
  """Analysis of collected samples."""
  tagged_samples = autotents.digits.manager.data
  flat_samples = sum(
    [[(tag, i, s)] for tag, samples in tagged_samples.items() for i, s in enumerate(samples)],
    []
  )
  print(f'Sample count is: {len(flat_samples)}.')
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
      val = autotents.common.rescale_and_match(s_img,s_pat, autotents.common.TM_METHOD)
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
  main_analyze_samples()
