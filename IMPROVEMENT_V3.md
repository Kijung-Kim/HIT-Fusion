# Improvement version 3

The submitted manuscript is unchanged. This is a separately versioned
development experiment, not an exact reproduction of the submitted method.
The v2 implementation and its measurements remain in Git history and at tag
`improvement-v2` (commit `0c40276`, with simplified version comments).

## Diagnosed causes and changes

1. Endpoint refinement moved five verified candidates outside their GT sets.
   The lowest pixel of a motion component is an indirect club proxy and can
   precede ball contact. v3 outputs the verified joint candidate without this
   refinement. This has a known tradeoff: M20191015164450204 was corrected by
   the old refinement, while its ball position was visibly displaced. v3
   exposes that late candidate rather than fixing its localization.
2. The previous stable search used offsets 0..19, consuming one of its 20
   positions on the candidate itself. P16 and P16_short have their third
   consecutive zero at forward offset 20. v3 searches actual neighbors at
   offsets 1..20, still respecting the candidate interval. It does not change
   the three-zero criterion, pixel thresholds, or overall search length to
   fit other failed clips. This differs from the explicit submitted rule.

## Isolated comparison

Replaying stored v2 candidate records on all clips, without changing image
processing or GT, gives 64/72 if endpoint refinement alone is removed versus
60/72 for v2: five corrected failures and one regression. This replay is an
ablation, not a separate end-to-end run. The full v3 run additionally tests
the actual-neighbor search and is reported below when complete.

GT definitions remain unchanged from the v2 experiment. The primary local
set has 72 annotated clips, with p33/p47 unscored and P16 labels retained
pending confirmation. The original Excel 74-clip result is reported separately.
These clips inform development, so improvements are not independent evidence
of generalization or proof of the manuscript's archived 72/74 result.

## Validation

Build and evaluation tests are retained. New rule assertions cover the 20th
neighbor, reject the 21st, respect sequence boundaries, and exclude the
candidate itself from a stable-neighbor run. Full execution reuses v2's
person segmentation masks and keeps all other preprocessing and thresholds.

## Completed full run

| Fixed GT set | v2 | v3 |
|---|---:|---:|
| User-corrected 72 (P16 pending) | 60/72 (83.33%) | 66/72 (91.67%) |
| Original Excel 74 | 60/74 | 66/74 |

Corrected failures: P16, P16_short, p35, M20191014151548416, M20191014153821811, M20191016122725247, M20191016155822876.
Regressions: M20191015164450204.
Remaining on the 72-clip set: 1 wrong-frame outputs and
5 abstentions. The lower bound on archived performance
reproduction is not resolved: this is a method variant evaluated on local
development clips, with historical model/corpus differences still open.
