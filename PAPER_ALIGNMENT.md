# Manuscript-to-code audit

Baseline: commit `15619148902a025e0338aab317ea1d335f681da4` on `master`.
Reference: submitted review PDF, submission ID
`71d3fe97-cd94-4ed9-b704-7e72ac540bd1`, generated 4 November 2026.

This is a reconstruction from the manuscript, not proof that the submitted
description was the implementation behind the archived numbers.

| Component | Public baseline | Manuscript / reconstructed behavior |
|---|---|---|
| Person occlusion | Chooses a small person box; no actual overlap rejection | Reject selected-player overlap with another detected box |
| Ball ROI side | Always right of player | Matched segmentation mask, lower-half pixel-center centroid; tie selects right |
| Missing person | Invalid sentinel and x-coordinate check | Explicit invalid detection, mask, and ROI rejection |
| Retinex | Computed, but extraction reads unenhanced channels | Enhanced channels actually feed blob extraction |
| Ball voting | Inverse-distance weights within blob radius | Normalized Gaussian over ball ROI; sigma=40/diameter |
| Candidate extrema / cuts | Sentinel overflow, integer gaps, inconsistent intervals | Exact positive sets, floating-point gaps/cut, one bounded interval |
| Cut ties | Descending scan approximates latest minimizer | Explicit latest minimizer; earliest eligible maximum motion |
| Stable frames | Unbounded unsigned indices; default zero can stand in for failure | Offsets 0..19 inside candidate interval, three actual zeros, explicit failure |
| Stable comparison | Later record's previous image initializes comparison toward earlier current image | Earlier stable current image initializes MOG2, later current image is compared |
| Endpoint mask | Starts at record 100; endpoint validity not initialized reliably | Every record with both neighboring images; largest component and explicit missing endpoints |
| Refinement | Window excludes +4; conditional +1 frame; no distance tie rule | Inclusive -4..+4, nearest then earliest tie, retain candidate if no endpoint |
| File IDs | Unsorted filesystem order and first-number offset | Sort numeric filename IDs and return actual selected filename |
| Ball candidates | Fixed-size array can be written at index 30 | Unbounded vector for all paper-path votes; legacy array remains safely bounded |
| Evaluation | No reproducible evaluation harness | Exact frame-set membership, explicit missed outputs, separate mixed records |

## Explicit reconstruction choices and limitations

- Original detector `.weights` bytes are recoverable through Git LFS. The
  segmentation checkpoint identity used historically is unknown. A newly
  downloaded official YOLOv8n-seg checkpoint is recorded by SHA-256.
- Segmentation confidence 0.25, full-resolution masks, and maximum-positive-IoU
  box matching are explicit choices where the manuscript omits exact details.
- The inherited person detector selects minimum box height as its definition
  of a smaller person, preserves its existing confidence convention, and uses
  detection boxes for overlap screening. This heuristic may reject bystanders
  or duplicate detections and is not a general person-identity tracker.
- Random seeds are fixed to 0. The manuscript does not specify the historical
  Retinex random seed. The enhanced image is now used; this intentionally changes
  the effective preprocessing from the public baseline.
- All extracted candidates contribute to the paper-path vote. The manuscript
  specifies no 30-candidate cap. The bounded legacy array is retained only for
  legacy consumers and does not determine the paper-path vote.
- No thresholds are fitted to GT and no annotations are inferred from folder
  names or from algorithm predictions.
- The local inventory selected by the documented camera-stream rule has 82
  sequences and 19,472 frames. The manuscript reports 82 sequences and 20,763
  frames. This difference must be reconciled before claiming the original
  evaluation corpus was reproduced.
- User-supplied Sheet1 column D provides the accepted frame sets for 74 clips;
  eight blank entries remain unscored. These sets are used without adding a
  further tolerance. Archived predictions and the historical segmentation
  checkpoint remain unavailable. See VISUAL_AUDIT.md for measured results.
- `pipeline_seconds` excludes Python segmentation and includes C++ model load,
  image loading, and processing. `segmentation_seconds` includes segmentation
  inference; first-use setup can affect it. `sequence_seconds` also includes
  cache writes and subprocess overhead. Concurrent batch timings are diagnostic,
  not a reproduction of the archived i7 desktop's 7.225 seconds per sequence.

## Sources

- [Original public implementation](https://github.com/Kijung-Kim/HIT-Fusion/tree/15619148902a025e0338aab317ea1d335f681da4)
- [Official Ultralytics segmentation API](https://docs.ultralytics.com/tasks/segment/)

## Before publishing a reproduced performance claim

Reconcile local corpus differences, recover the historical model and predictions,
compare all per-sequence predictions and abstentions, and retain the model
hashes and run configuration. A target aggregate number is not evidence of
algorithmic equivalence or a reason to change the annotations.
