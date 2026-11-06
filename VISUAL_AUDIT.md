# Visual audit and measured reproduction (2026-11-06)

The paper-path reconstruction was run again on all 82 local sequences after
removing the undocumented 30-candidate voting cap. No thresholds or GT sets
were changed. The supplied Sheet1 column D contains accepted frame sets for
74 clips; the eight blank entries are unscored, not verified true negatives.

| Metric | Before cap removal | After cap removal |
|---|---:|---:|
| Correct / annotated | 52 / 74 | 52 / 74 |
| Localization success | 70.27% | 70.27% |
| Precision | 91.23% | 91.23% |
| Recall | 70.27% | 70.27% |
| F1 | 79.39% | 79.39% |

This comparison is against the previous manuscript reconstruction, not a
fresh measurement of the original public executable. The archived paper
result of 72/74 (97.30%) has **not** been reproduced.

The maximum observed per-frame candidate count was 8.
There were 0 frames above the old cap and
0 changed outputs/statuses. Removing the cap fixes
an undocumented implementation constraint; it is not evidence of improved
accuracy on this corpus.

Visual inspection of original frames and execution logs found:

- Incorrect ball locations on grass: p18, p20, p21, p23,
  M20191014151428299, M20191014151548416, M20191014160641664,
  M20191014160740950. The two practice_gate failures contain visible hits.
  That status means low change near the estimated ball, not proven practice.
- Estimated positions below the ball and failed stable comparison: p19,
  p42_green, M20191015164526538. All three selected a GT frame first;
  the stable comparison returned zero foreground pixels.
- Missing post-candidate stable neighbors: P16 and P16_short selected 1593
  correctly but were rejected. M20191015164256784 selected 580, outside its
  accepted 578/579 set, and also failed the stable-neighbor check.
- Foreground people obscure the player in the initialization frame:
  p33_occlusion (invalid ROI), p44 and M20191016122611875 (person overlap),
  p47 (invalid person). The exact incorrect box in p33 was not isolated.
- Endpoint refinement moved four correct candidates one frame early:
  p35 472->471, M20191014153821811 2143->2142,
  M20191016122725247 789->788, M20191016155822876 257->256.

These observations identify failure stages, not a full causal ablation of
every preprocessing choice. Loosening the stable rules or removing endpoint
refinement would change the manuscript method and has not been done.
Historical segmentation weights, archived intermediate results, and the
19,472-versus-20,763 frame inventory discrepancy remain unresolved.

Validation: MSVC/OpenCV 3.4.0 build, C++ rule assertions, two evaluation unit
tests, and the complete 82-sequence execution. Concurrent timings are not
comparable to the archived single-machine runtime claim. Stage diagnostics
are emitted as AUDIT lines without affecting the decision rules.

Private frames, clips, GT and execution evidence are retained outside this
repository. No remote publication was performed as part of this audit.
