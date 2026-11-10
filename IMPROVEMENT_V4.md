# Improvement version 4

The submitted paper and original annotations are unchanged. This is a new
development method, not proof that the archived manuscript method is reproduced.
Version 3 is preserved at tag `improvement-v3`.

## Changes and their evidence

### Ball localization

Raw white-balanced and Retinex-enhanced channels both supply blob candidates.
Existing area, convexity, separation and integer-coordinate rules are retained.
Every Gaussian kernel still has center value 1 and sigma=40/diameter.

The accumulated vote map is ranked with a separate appearance-support map.
The reference is the per-channel median of the three original-color ball-ROI
images at offsets -7,-6,-5 from maximum player-motion energy (clamped at zero).
A 3x3 elliptical opening suppresses pixel noise; a 19x19 elliptical top-hat
extracts compact bright support, taking the maximum response over RGB channels.
The 19-pixel scale covers the existing maximum original-coordinate blob diameter
(about 17 pixels) plus margin. The final spatial score is vote times appearance;
if that product has no positive support, voting alone is retained.

Unit kernel peaks do not mean the final appearance-modulated map has unit peaks.
No diameter-squared center multiplier is reintroduced. The reference uses image
motion only, not GT. Initialization-frame appearance was rejected because the
addressing club could contaminate it. A trial circularity cutoff of 0.7 rejected
real balls and was removed; it is not in v4.

### Partially occluded players

Overlapping person bounding boxes no longer cause unconditional rejection.
The selected player still needs a valid matched segmentation mask and ROI.
Other cached person-instance masks suppress covered candidate centers and vote
pixels. The ball-change and stable-state checks remain mandatory. These masks
come from the initialization frame; this does not solve arbitrary moving
occluders or prove that every pixel of a ball patch is visible.

### Temporal verification and output

The existing clip-adaptive eligibility cuts remain. Eligible candidates are
ranked by local-change times motion energy, rather than motion energy alone.
Stable frames require three consecutive local-change counts <=13 within the
same 20-neighbor search. 13 is reused from the existing final state-change
threshold; missing scores are never stable. The final before/after comparison
still must exceed 13 pixels.

After verifying that event, output is its local rising edge: scan at most four
frames backward through a contiguous run with local change at least half the
verified anchor's change. Stop at the first lower or missing score. This is
an integer-only postprocessing rule, unrelated to the removed motion-mask
endpoint proxy. The verified anchor is retained separately in output metadata.

## Validation provenance

All 82 clips are processed through the new localization and event verification.
The final integer onset rule is replayed using the production C++ function on
the retained local-change scores for all accepted events. The six target clips
and three prior correct controls are additionally reprocessed with the final
binary, checking exact agreement with the replayed results. Separate runs on
the six targets isolate ball, temporal and occlusion changes; three v3-profile
runs check compatibility with archived v3 outputs.

The preliminary ball/temporal/occlusion ablations precede onset postprocessing
and are not full-corpus accuracy measurements. Their files and the original
verified-peak predictions are retained outside the source repository. Ground
truth is used only by the evaluator and never provided to inference.

The 72-clip GT definition excludes p33/p47 per the user and retains P16 labels
pending clarification. The original Excel 74-clip definition is also scored.
The remaining ten clips are unscored, not automatically true negatives. This
corpus informed development; its score is not independent generalization
evidence. Historical model identity and the 19,472/20,763 frame discrepancy
remain unresolved. Concurrent runtimes are not comparable to the paper's timing.

## Run profiles

Use `python tools/reproduce.py ... --profile v4` for the default revision.
`v3`, `v4_ball`, `v4_temporal`, and `v4_occlusion` permit controlled comparisons.
The final `v4_temporal` profile includes onset; archived preliminary ablations
without onset are identified in their provenance. Full results are added below
after verification. No paper edit, remote push, or editor contact is performed.

## Completed comparison

| Fixed GT definition | v3 | v4 |
|---|---:|---:|
| User-corrected 72, P16 pending | 66/72 (91.67%) | 72/72 (100.00%) |
| Original Excel 74 | 66/74 | 72/74 (97.30%) |

Corrected cases: p44, M20191014151428299, M20191015164256784, M20191015164450204, M20191015164526538, M20191016122611875. Regressions: none.
The 72-clip set has 0 wrong-frame outputs and 0
abstentions. All 10 unscored clips remain outside the accuracy denominator;
10 have no output. A numerical match to the archived percentage
is not evidence of reproducing the original method or historical experiment.

The nine final-binary image reruns exactly matched the production-C++ onset
replay, including ball coordinates, anchor, output index and frame ID. Three
v3 compatibility runs matched archived v3. The source and all version history
are retained locally; the paper and remote repository are not modified.
