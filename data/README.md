# Training data

Each CSV in this directory is one exercise. The filename (without `.csv`)
is the model key. `train.py` picks up every `*.csv` here automatically; the
live UI uses `exercise_names.json` for its display name.

`exercise_names.json` provides display names without changing the model keys.
The current names are hypotheses inferred from the saved landmark patterns;
confirm them against the original recordings before presenting them as facts.
Edit this JSON file to correct a name.

## Format

One row per collected sample:

| column | meaning |
|---|---|
| `label` | form-quality class (see below) |
| `x0,y0,z0 … x32,y32,z32` | 33 MediaPipe pose landmarks, normalized image coordinates |

## Classes

| class | meaning |
|---|---|
| `Up` | top of the movement |
| `Down` | bottom of the movement |
| `Optimal` | correct form |
| `subOptimal` | form breaking down — warning |
| `Dangerous` | injury-risk form — alert |

## Collection protocol

Samples were captured with `python -m formcoach.collect` from a laptop webcam:
the subject performs the movement while an operator presses the class key at
the matching moment. Each keypress stores one pose snapshot. Datasets here
contain ~520–640 samples per exercise, roughly balanced across the 5 classes.

To collect a new exercise:

```bash
python -m formcoach.collect --out data/lunge.csv
python -m formcoach.train        # retrains everything, including the new file
```
