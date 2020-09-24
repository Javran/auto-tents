# Automation of "tents & trees" Game

Note: you will need to manually create `private/` directory.

Requirement:

- an Android phone.
- opencv-python and python 3.7. I have no idea nor do I care why it only works on 3.7.
- adb, we will do screenshots and issue taps using it.
- a compiled binary of [tents-demo](https://github.com/Javran/puzzle-solving-collection/tree/master/tents-solver).

Here are few steps to have a working program, if my memory serves.

- Collect sample for different sizes.

  Each different puzzle size needs one.
  Collect and store them under `private/samples/{h}x{w}/`,
  where `h` is the screen height of the phone and `w` screen width.
  The naming pattern is `sample-{x}-{x}.png` where `{x}` is
  the size of the puzzle (a number from `5` to `22`).

  To take phone screenshots, use `adb exec-out screencap -p > sample-{size}x{size}.png`.

- Generate `private/preset.json`.

  This file allows us to quickly find cell positions
  for all different sizes of the puzzle board.

  For this you just need to run `cd py; ./gen_preset.py`.

- Collect digit samples.

  We'll need to visit through those samples again
  in order to collect digit samples for recognition to work.
  Tagging with `cd py; ./tagging.py`.

- Have a working [tents-demo](https://github.com/Javran/puzzle-solving-collection/tree/master/tents-solver) binary.

- Set environment variable `TENTS_DEMO_BIN` to the location of the binary. `cd py/` then `./solver.py` when the phone is at a game screen.

- (Optional) Set environment variable `PUZZLE_RECORDS` to a file path to append recognized puzzles to it.

- `cd py/; ./analyze_samples.py` can used to gather some analysis,
  this is mostly just for experimenting with threshold methods.
