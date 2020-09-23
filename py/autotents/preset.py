import json
import os

import autotents.common

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


preset = Preset()
