#!/usr/bin/env python3.7

import json
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid

import cv2
import numpy
import input_agent_client

import autotents.common
import autotents.digits
import autotents.preset


def get_tents_demo_bin():
  """Gets path to compiled binary `tents-demo`. (See README.md for detail)."""
  tents_demo_bin = os.environ['TENTS_DEMO_BIN']
  assert os.path.exists(tents_demo_bin)
  return tents_demo_bin


def load_realtime_screenshot(aia_client):
  img_data = aia_client.commandScreenshotAll()
  img_np = numpy.frombuffer(img_data, dtype=numpy.uint8)
  img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
  return img

def main_recognize_and_solve_board():
  tents_demo_bin = get_tents_demo_bin()
  print(f'tents-demo: {tents_demo_bin}')

  if 'AIA_PORT' not in os.environ:
    print('AIA_PORT is not set.')
    sys.exit(1)

  aia_client = input_agent_client.InputAgentClient(int(os.environ['AIA_PORT']))

  # take screenshot
  img = load_realtime_screenshot(aia_client)
  h, w, _ = img.shape
  # pick preset and determine screen_dim and size.
  screen_dim = (h, w)
  screen_dim_raw = f'{h}x{w}'
  assert screen_dim_raw in autotents.preset.preset.data, \
    f'Current preset does not contain info about screen size {screen_dim_raw}.'
  size = autotents.preset.preset.findBoardSize(img, screen_dim)
  assert size is not None, 'Size cannot be recognized.'
  print(f'Board size: {size}x{size}')
  cell_bounds = autotents.preset.preset.getCellBounds(size, screen_dim)
  row_bounds, col_bounds = cell_bounds
  row_digits, col_digits = autotents.common.extract_digits(img, cell_bounds)

  digits = numpy.concatenate(
    [
      numpy.concatenate(row_digits, axis=1),
      numpy.concatenate(col_digits, axis=1),
    ])

  cells = [ [ None for _ in range(size) ] for _ in range(size)]
  for r, (row_lo, row_hi) in enumerate(row_bounds):
    for c, (col_lo, col_hi) in enumerate(col_bounds):
      cells[r][c] = img[row_lo:row_hi+1, col_lo:col_hi+1]
  recombined = numpy.concatenate([ numpy.concatenate(row, axis=1) for row in cells ], axis=0)

  output_board = [ [ None for _ in range(size) ] for _ in range(size)]
  def find_tree(cell_img,r,c):
    result = autotents.common.find_exact_color(
      cell_img, autotents.common.COLOR_TREE_SHADE)
    (_,_,w,h) = cv2.boundingRect(result)
    if w != 0 and h != 0:
      color = 0xFF
      output_board[r][c] = 'R'
    else:
      color = 0
      output_board[r][c] = '?'
    return numpy.full((4,4), color)

  cell_results_recombined = numpy.concatenate([
    numpy.concatenate([
      find_tree(cell,r,c)
      for c, cell in enumerate(row)
    ], axis=1)
    for r, row in enumerate(cells)
  ], axis=0)

  # tagged_samples = load_samples()
  recog_row_digits = [ None for _ in range(size) ]
  recog_col_digits = [ None for _ in range(size) ]

  confident = True
  for desc, ds, ds_out in [
      ('Row', row_digits, recog_row_digits),
      ('Col', col_digits, recog_col_digits),
  ]:
    for i, digit_img in enumerate(ds):
      need_to_save = False
      digit_img_cropped = autotents.common.crop_digit_cell(digit_img)
      if digit_img_cropped is None:
        ds_out[i] = '0'
        continue
      # use original image for this step as we want some room around
      # the sample to allow some flexibility.
      best_val, best_tag, competing_factor = autotents.digits.manager.findTag(digit_img)
      if best_val is None or best_val < autotents.common.RECOG_THRESHOLD:
        confident = False
        need_to_save = True
        print(f'Warning: best_val is only {best_val}, the recognized digit might be incorrect.')
      if competing_factor is not None:
        confident = False
        need_to_save = True
        print(f'Warning: found a competing factor of {competing_factor}, proceed to sampling.')
      if need_to_save:
        nonce = str(uuid.uuid4())
        if best_val is None:
          fname = f'UNTAGGED_{nonce}.png'
        else:
          print(f'Found new sample with best guess being {best_tag}, with score {best_val}')
          # attach the suspected tag here so it is more convenient when it is actually correct.
          fname = f'UNTAGGED_{best_tag}_{nonce}.png'
        store_path = autotents.common.private_path('digits')
        fpath = os.path.join(store_path, fname)
        print(f'Saving a sample shaped {digit_img_cropped.shape} to {fpath}...')
        cv2.imwrite(fpath, digit_img_cropped)

      ds_out[i] = best_tag
  assert confident, 'Solving process stopped as recognition might be inaccurate.'
  # Recognition is done, build up input to tents-demo
  input_lines = []
  def out(line):
    input_lines.append(line)

  out(f'{size} {size}')
  for i, line in enumerate(output_board):
    input_lines
    out(''.join(line) + f' {recog_row_digits[i]}')
  out(' '.join(recog_col_digits))
  print('# PUZZLE OUTPUT BEGIN')
  for l in input_lines:
    print(l)
  print('# PUZZLE OUTPUT END')

  if confident and 'PUZZLE_RECORDS' in os.environ:
    puzzle_file = os.environ['PUZZLE_RECORDS']
    with open(puzzle_file, 'a') as f:
      print(f'# {uuid.uuid4()}', file=f)
      for l in input_lines:
        print(l, file=f)
    print(f'Recorded to {puzzle_file}.')
  skip_solving = False
  plot = False
  if plot:
    pyplot.figure().canvas.set_window_title('@dev')
    subplot_color(221, img, 'origin')
    subplot_color(222, recombined, 'extracted')
    subplot_color(223, digits, 'digits')
    subplot_gray(224, cell_results_recombined, 'find tree')
    pyplot.show()
  if skip_solving:
    return
  proc_result = subprocess.run(
    [tents_demo_bin, 'stdin'],
    input='\n'.join(input_lines) + '\n',
    text=True,
    capture_output=True,
  )
  raw_tent_positions = proc_result.stdout.strip().split('|')
  def parse_raw(raw):
    [a,b] = raw.split(',')
    return int(a), int(b)
  tent_positions = list(map(parse_raw, raw_tent_positions))
  print(f'Received {len(tent_positions)} tent positions.')
  # puzzle is solved, build up plan to tap cells as necessary

  def tap(r,c):
    row_lo, row_hi = row_bounds[r]
    row_pos = round((row_lo + row_hi) / 2)
    col_lo, col_hi = col_bounds[c]
    col_pos = round((col_lo + col_hi) / 2)
    coord = (col_pos, row_pos)
    aia_client.commandTap(coord)

  solving_moves = [ d for pos in tent_positions for d in [pos, pos] ]
  # shuffling doesn't actually do much, but looks a bit fancier.
  random.shuffle(solving_moves)
  for (r,c) in solving_moves:
    tap(r,c)
    time.sleep(0.02)


if __name__ == '__main__':
  main_recognize_and_solve_board()
