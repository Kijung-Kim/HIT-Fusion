# Reproduce the submitted manuscript

The default Visual Studio entry point is now `paper_main.cpp`, a headless,
manuscript-aligned reconstruction. The old GUI driver remains in `golfBall.cpp`
for source history. The historical numerical results have **not** been verified
for this reconstruction. Do not quote 97.30% as a new result without evaluating
against the original frame-set annotations.

## Build

Requirements: Visual Studio 2019 C++ tools (v142), x64 OpenCV **3.4.0**, Python
3.10+ for segmentation and experiment orchestration. OpenCV 3.4.0 is deliberately
retained for the original Darknet parser, MOG2, and Farneback implementation.

```powershell
git lfs pull
$env:OpenCV_DIR = 'C:\opencv\opencv340\build'
.\build_paper.ps1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduce.txt
```

Alternatively, build `golfBall.sln` in x64 Release. This produces `golfBall.exe`
with the same paper entry point. `build_paper.ps1` produces `hit_paper.exe` for
the Python driver.

`yolo.cfg` and the Git LFS `yolo.weights` must be present in the repository root.
A 134-byte LFS pointer is not a usable checkpoint. The Python driver downloads
the official `yolov8n-seg.pt` when the requested file is missing, and records its
SHA-256. A newly downloaded checkpoint does not establish identity with the
unavailable historical segmentation model.

## Run and retain evidence

```powershell
.\.venv\Scripts\python.exe tools/reproduce.py `
  --data C:\researches\golfSwing\sq `
  --output reproduction\run `
  --model reproduction\models\yolov8n-seg.pt
```

The driver chooses the original `506cbd` camera stream (or flattened JPEG
folders), sorts frames by their numeric filename, and caches person instance
masks from stored frame 10. Segmentation uses CPU, input size 640, confidence
0.25, and masks restored to the original image resolution. The segmentation
box with the largest positive IoU to the Darknet player box supplies the mask.
Mask confidence and matching details were not fully specified in the manuscript;
these are explicit reconstruction choices, not recovered historical settings.

Each sequence produces its predicted **numeric filename frame ID**, a reason
when no impact is returned, raw signals, and timing. Predictions use actual
filenames rather than adding the first filename to a stored index. Masks and
raw frames remain local; do not commit participant images.

The direct executable accepts:

```text
hit_paper.exe FRAMES_DIR SEGMENTATION.yml RESULT.yml SIGNALS.csv
```

Run it with the repository root as working directory and OpenCV DLLs on PATH.
The segmentation cache is produced by `tools/reproduce.py`. Rejections are
abstentions, not verified non-impact labels. `status: error` indicates a failed
run and must be resolved before reporting accuracy.

## Evaluate

Supply a JSON object keyed by the same sequence IDs as the run. Frame numbers
are integer filename stems. Example schema (illustrative, not real annotations):

```json
{
  "clip_a": {"kind": "impact", "frames": [143, 144]},
  "clip_b": {"kind": "mixed_or_indeterminate"},
  "clip_c": {"kind": "verified_nonimpact"}
}
```

Add `--gt PATH_TO_GT.json` to the run command. Every run sequence must have a GT
entry; mismatches fail rather than changing the denominator. Success is exact
membership in the annotated frame set, with **no extra temporal tolerance**.
A wrong-frame prediction counts as both FP and FN. Event precision, recall,
and F1 are computed on the annotated-impact subset, matching manuscript Table
III. Verified negatives and mixed/indeterminate records are reported separately.
The mixed eight records must not be automatically labeled true negatives.

To evaluate previously saved predictions without running the videos again:

```powershell
python tools/evaluate.py --predictions reproduction/run/predictions.json `
  --gt original_gt.json --output reproduction/metrics.json
```

For a four-process execution use `--shards 4 --shard-index 0` through `3`, each
with a separate output directory. Throughput and timing from concurrent runs
must not be presented as the manuscript's archived single-sequence CPU timing.

## Tests

```powershell
python -m unittest discover -s tests -p test_evaluation.py
g++ -std=c++11 tests/paper_rules_test.cpp -o paper_rules_test.exe
.\paper_rules_test.exe
```

The C++ test covers fractional thresholds, tie rules, unavailable scores,
bounded stable-frame searches, the inclusive +4 refinement boundary, and
missing endpoints. The evaluation tests check wrong-frame accounting and
prevent silent sequence omission.
